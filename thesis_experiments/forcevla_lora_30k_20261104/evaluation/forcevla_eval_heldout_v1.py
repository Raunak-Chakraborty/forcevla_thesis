#!/usr/bin/env python3
"""Evaluation-only, read-only, pilot-compatible ForceVLA LoRA checkpoint evaluator.

Only uses the old, 2026-10-30, working midpoint validation protocol:
34 validation episodes 255..288; option to use first six; seed=42000+episode.
Neither imports nor invokes training loops, checkpoint writers, or robots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path('/home/ktt_rc/robotics_thesis/forcevla_lab')
WT = ROOT / 'worktrees/forcevla_lora_reliability_v1'
DATA = Path('/mnt/robotics_ssd/ktt_rc_robotics_thesis/dataset')
CK_BASE = DATA / 'forcevla_reliable_runs/checkpoints/doosan_forcevla_25d_32d_compat_contract/LORA_5H_20261104'
RUN = ROOT / 'logs/lora_reliability/runs/LORA_5H_20261104'
BASELINE = ROOT / 'logs/finetune_readiness/force_eval_6ep_step1999.json'
EXPECTED_COMMIT = 'cdf4677244513a52665a16f5319f662379ebe9bc'
SOURCE_HASHES = {
    'forcevla_offline_eval_v1.py': '30bf5d9ce8576a7cc29eda2916a6935e29495a1eb144ffe0c61e3c944aec64da',
    'forcevla_validation_bridge_v1.py': '8d36132524e11d9d86c8da03d1b80a03158c73d32046e0de5ed48421b930eda2',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_before_import(step):
    import subprocess
    commit = subprocess.check_output(['git', '-C', str(WT), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f'Wrong source commit: {commit}')
    if subprocess.check_output(['git', '-C', str(WT), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Reliability worktree dirty; checkpoint/source provenance unclear')
    for name, expected in SOURCE_HASHES.items():
        src = ROOT / 'scripts' / name
        if not src.is_file() or digest(src) != expected:
            raise RuntimeError(f'Existing evaluator source differs from approved ZIP: {src}')
    manifest = json.loads((RUN / 'scientific_contract.json').read_text())
    if manifest.get('experiment') != 'LORA_5H_20261104' or manifest.get('target_updates') != 30000:
        raise RuntimeError('Training scientific contract changed')
    ck = CK_BASE / str(step)
    receipt_path = RUN / 'receipts' / f'step-{step:08d}.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('schema') != 'forcevla-checksummed-orbax-v1' or receipt.get('completed_updates') != step or receipt.get('folder') != str(step):
        raise RuntimeError('Checkpoint receipt contract mismatch')
    if not (ck / '_CHECKPOINT_METADATA').is_file() or not (ck / 'params').is_dir() or not (ck / 'train_state').is_dir():
        raise RuntimeError('Orbax checkpoint incomplete')
    files = {p.relative_to(ck).as_posix() for p in ck.rglob('*') if p.is_file()}
    if files != set(receipt['files']):
        raise RuntimeError('Checkpoint file inventory differs from receipt')
    for rel, entry in receipt['files'].items():
        p = ck / rel
        if p.is_symlink() or p.stat().st_size != entry['size']:
            raise RuntimeError(f'Checkpoint file missing or changed: {p}')
    if step == 30000:
        latest = json.loads((RUN / 'latest_committed.json').read_text())
        if latest != receipt:
            raise RuntimeError('Final checkpoint receipt is not the authoritative latest pointer')
    print(f'CHECKPOINT_RECEIPT=PASS STEP={step} FILES={len(files)}', flush=True)
    print('NOTE=Size/inventory verified; this pilot does not reread all checkpoint SHA256 bytes', flush=True)
    return ck, digest(receipt_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--step', type=int, choices=[5000, 10000, 20000, 30000], required=True)
    ap.add_argument('--episodes', type=int, choices=[6, 34], required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refuse overwrite: {args.output}')
    if not args.output.parent.is_dir():
        raise FileNotFoundError(f'Output directory missing: {args.output.parent}')
    ck, receipt_hash = verify_before_import(args.step)
    # Existing evaluator already contains the validated float32 image conversion.
    # Load those helpers unchanged; do not copy or fork their normalization logic.
    sys.path.insert(0, str(ROOT / 'scripts'))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from openpi.models import model as model_lib
    from openpi.shared import nnx_utils
    from forcevla_validation_bridge_v1 import load_validation, check_sample
    from forcevla_offline_eval_v1 import make_observation

    if len(jax.devices()) != 1 or jax.devices()[0].platform != 'gpu':
        raise RuntimeError(f'Expected exactly one visible GPU, got {jax.devices()}')
    print('GPU_EVALUATION_DEVICE=PASS', flush=True)
    print('MODEL_TRAINING=DISABLED', flush=True)

    t0 = time.monotonic()
    transformed, indices, config = load_validation()
    selected = list(indices.items())[:args.episodes]
    expected_eps = list(range(255, 255 + args.episodes))
    if [int(ep) for ep, _ in selected] != expected_eps:
        raise RuntimeError('Validation split ID/order mismatch')
    print(f'VALIDATION_ISOLATION=PASS EPISODES={args.episodes}', flush=True)
    print('DATASET=COMPLETE_323_EPISODE_SPLIT; TRAIN_NORMALIZATION_REUSED=YES', flush=True)

    params = model_lib.restore_params(ck / 'params')
    model = config.model.load(params)
    model.eval()
    inference = nnx_utils.module_jit(model.sample_actions)
    print(f'MODEL_RESTORE=PASS CHECKPOINT_STEP={args.step}', flush=True)

    results = []
    for number, (episode, index) in enumerate(selected, 1):
        sample = transformed[index]
        _, actions = check_sample(sample)
        observation = make_observation(sample)
        if tuple(np.asarray(observation.state).shape) != (1, 25):
            raise RuntimeError('Invalid normalized input state shape')
        no_force_observation = observation.replace(state=observation.state.at[:, 19:25].set(0.0))
        key = jax.random.key(42000 + episode)  # identical per-episode seed as Oct-30 pilot
        t = time.monotonic()
        actual = np.asarray(jax.device_get(inference(key, observation)))
        masked = np.asarray(jax.device_get(inference(key, no_force_observation)))
        if actual.shape != (1, 50, 32) or masked.shape != (1, 50, 32):
            raise RuntimeError(f'Wrong action shape: {actual.shape} and {masked.shape}')
        if not np.isfinite(actual).all() or not np.isfinite(masked).all():
            raise RuntimeError('Nonfinite action predictions')
        truth = np.asarray(actions)[:, :7]
        prediction, baseline = actual[0, :, :7], masked[0, :, :7]
        diff = np.abs(prediction - truth)
        noforce_diff = np.abs(baseline - truth)
        result = {
            'episode': int(episode),
            'frame_index': int(index),
            'mae_actual': float(diff.mean()),
            'mae_masked': float(noforce_diff.mean()),
            'force_sensitivity': float(np.abs(prediction - baseline).mean()),
            'first_action_sensitivity': float(np.abs(prediction[0] - baseline[0]).mean()),
            'first_action_mae_actual': float(diff[0].mean()),
            'first_action_mae_masked': float(noforce_diff[0].mean()),
            'mae_per_semantic_dimension': [float(x) for x in diff.mean(axis=0)],
            'seconds': round(time.monotonic() - t, 3),
        }
        if not all(math.isfinite(result[k]) for k in ('mae_actual','mae_masked','force_sensitivity','first_action_sensitivity','first_action_mae_actual','first_action_mae_masked')):
            raise RuntimeError('Nonfinite metric')
        results.append(result)
        print('EPISODE_RESULT=' + json.dumps({k: v for k, v in result.items() if k != 'mae_per_semantic_dimension'}, sort_keys=True), flush=True)
        print(f'EPISODES_COMPLETED={number}/{args.episodes}', flush=True)

    def mean(k):
        return round(sum(x[k] for x in results) / len(results), 6)
    summary = {
        'checkpoint_step': args.step,
        'episodes_evaluated': args.episodes,
        'semantic_action_dims': 7,
        'mae_actual': mean('mae_actual'),
        'mae_masked': mean('mae_masked'),
        'force_sensitivity': mean('force_sensitivity'),
        'first_action_sensitivity': mean('first_action_sensitivity'),
        'first_action_mae_actual': mean('first_action_mae_actual'),
        'first_action_mae_masked': mean('first_action_mae_masked'),
        'mae_per_semantic_dimension': [round(sum(row['mae_per_semantic_dimension'][i] for row in results) / len(results), 6) for i in range(7)],
        'recorded_force_better_than_masked': mean('mae_actual') < mean('mae_masked'),
        'inference_only': True,
        'validation_only': True,
        'normalization_units': 'TRAINING_NORMALIZED_ACTIONS',
        'checkpoint_receipt_sha256': receipt_hash,
        'evaluator_source_sha256': SOURCE_HASHES,
        'seconds_total': round(time.monotonic() - t0, 3),
        'metric_note': 'Open-loop normalized 50x7 action MAE on one midpoint per episode; no robot task-success claim',
    }
    if args.episodes == 6 and BASELINE.is_file():
        old = json.loads(BASELINE.read_text())
        ids = [x['episode'] for x in old['episodes']]
        if ids != [x['episode'] for x in results] or old['summary']['episodes_evaluated'] != 6:
            raise RuntimeError('Old pilot comparison episode selection does not match')
        old_value = float(old['summary']['mae_actual'])
        summary['pilot_2000_step_mae_actual'] = old_value
        summary['delta_mae_vs_pilot'] = round(summary['mae_actual'] - old_value, 6)
        summary['improvement_percent_vs_pilot'] = round(100 * (old_value-summary['mae_actual']) / old_value, 3)
    elif args.episodes == 34:
        old34 = ROOT / 'logs/finetune_readiness/force_eval_34ep_step1999.json'
        if old34.is_file():
            old = json.loads(old34.read_text())
            if [x['episode'] for x in old['episodes']] == [x['episode'] for x in results]:
                summary['pilot_2000_step_mae_actual'] = float(old['summary']['mae_actual'])
                summary['delta_mae_vs_pilot'] = round(summary['mae_actual'] - summary['pilot_2000_step_mae_actual'], 6)

    payload = {'summary': summary, 'episodes': results}
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write('\n')
    print('EVAL_SUMMARY=' + json.dumps(summary, sort_keys=True), flush=True)
    print(f'EVAL_RESULTS={args.output}', flush=True)
    print('EVALUATION=PASS', flush=True)
    print('TRAINING_STARTED=NO', flush=True)


if __name__ == '__main__':
    main()
