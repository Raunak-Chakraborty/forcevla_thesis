"""Versioned reliability entrypoint; LoRA model/optimizer delegated to existing ForceVLA code."""
from __future__ import annotations
import argparse
import dataclasses
import fcntl
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import sys
import tempfile
import time
from datetime import datetime, timezone

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import replace
from openpi.training import checkpoints, sharding
from openpi.training.forcevla_replay import ReplayContract, validate_cursor
from openpi.training.forcevla_reliable_loader import create_replay_data_loader

ROOT = Path('/home/ktt_rc/robotics_thesis/forcevla_lab')
WT = ROOT / 'worktrees/forcevla_lora_100k_v1'
DATA = Path('/mnt/robotics_ssd/ktt_rc_robotics_thesis/dataset')
LOGROOT = ROOT / 'logs/lora_100k/runs'
INFO = DATA / 'forcevla_lerobot_language_v2_25d_train/meta/info.json'
NORM = ROOT / 'finetune_runs/assets/doosan_forcevla_25d_32d_compat_contract/forcevla_lerobot_language_v2_25d_train/norm_stats.json'
PILOT = WT / 'thesis_experiments/forcevla_lora_pilot_20261030/scripts/forcevla_lora_mock.py'
BASE_TRAIN = WT / 'scripts/train.py'
_STOP = False


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data, *, exclusive=False):
    path = Path(path)
    if path.is_symlink():
        raise RuntimeError(f'Unsafe symlink: {path}')
    fd, temp = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        if exclusive:
            os.link(temp, path)
            os.unlink(temp)
        else:
            os.replace(temp, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def checkpoint_files(directory):
    directory = Path(directory)
    if not (directory / '_CHECKPOINT_METADATA').is_file():
        raise RuntimeError(f'Orbax final metadata missing: {directory}')
    if not (directory / 'params').is_dir() or not (directory / 'train_state').is_dir():
        raise RuntimeError('Orbax checkpoint is missing params or train_state')
    rows = {}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise RuntimeError(f'Symlink inside checkpoint: {path}')
        if path.is_file():
            rel = path.relative_to(directory).as_posix()
            rows[rel] = {'size': path.stat().st_size, 'sha256': sha(path)}
    if not rows:
        raise RuntimeError('Empty checkpoint')
    return rows


def validate_receipt(directory, receipt):
    if receipt.get('schema') != 'forcevla-checksummed-orbax-v1':
        raise RuntimeError('Unrecognized checkpoint receipt schema')
    if str(receipt['folder']) != Path(directory).name:
        raise RuntimeError('Checkpoint directory/receipt mismatch')
    if receipt['files'] != checkpoint_files(directory):
        raise RuntimeError('Checkpoint integrity mismatch')


def publish_committed(manager, ckpt_dir, run_dir, completed):
    # This verifies Orbax completion and all payload bytes. It is NOT
    # a substitute for an independent full-state restore on a fresh process.
    manager.wait_until_finished()
    if completed not in set(map(int, manager.all_steps())):
        raise RuntimeError(f'Orbax step {completed} not committed')
    folder = Path(ckpt_dir) / str(completed)
    receipt = {
        'schema': 'forcevla-checksummed-orbax-v1',
        'folder': str(completed),
        'completed_updates': int(completed),
        'restore_tested': False,
        'time_utc': stamp(),
        'files': checkpoint_files(folder),
    }
    receipts = Path(run_dir) / 'receipts'
    receipts.mkdir(exist_ok=True)
    atomic_json(receipts / f'step-{completed:08d}.json', receipt, exclusive=True)
    atomic_json(Path(run_dir) / 'latest_committed.json', receipt)
    return receipt


def log_event(path, **fields):
    row = {'utc': stamp(), **fields}
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, sort_keys=True) + '\n')
        f.flush()
        os.fsync(f.fileno())


def _request_stop(signum, frame):
    global _STOP
    _STOP = True


def load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load source: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(args):
    pilot = load_source(PILOT, 'forcevla_baseline_pilot')
    baseline = pilot.make_config(args.experiment, args.steps)
    cfg = replace(
        baseline,
        lr_schedule=replace(baseline.lr_schedule, decay_steps=args.steps),
        checkpoint_base_dir=str(DATA / 'forcevla_100k_runs/checkpoints'),
        fsdp_devices=1, batch_size=2, num_workers=args.workers,
        log_interval=1, save_interval=args.save_every, keep_period=None,
        wandb_enabled=False, overwrite=False, resume=args.mode == 'resume',
    )
    info = json.loads(INFO.read_text())
    if info['total_frames'] != 289703 or info['total_episodes'] != 255:
        raise RuntimeError('Training dataset split changed')
    if (cfg.model.state_dim, cfg.model.action_dim, cfg.model.action_horizon) != (25, 32, 50):
        raise RuntimeError('ForceVLA science contract changed')
    if cfg.lr_schedule.decay_steps != 100000:
        raise RuntimeError('Expected 100000-step cosine schedule')
    if (cfg.batch_size, cfg.fsdp_devices, cfg.ema_decay) != (2, 1, None):
        raise RuntimeError('ForceVLA optimizer/parallelism contract changed')
    contract = ReplayContract(info['total_frames'], 2, cfg.seed, sha(INFO), sha(NORM))
    files = [
        BASE_TRAIN,
        WT / 'src/openpi/training/checkpoints.py',
        WT / 'src/openpi/training/data_loader.py',
        WT / 'src/openpi/training/forcevla_replay.py',
        WT / 'src/openpi/training/forcevla_reliable_loader.py',
        WT / 'src/openpi/models/pi0_force.py',
        WT / 'src/openpi/models/lora.py',
        PILOT,
        Path(__file__),
    ]
    manifest = {
        'schema': 'forcevla-reliable-science-v1',
        'experiment': args.experiment,
        'target_updates': args.steps,
        'batch_size': cfg.batch_size,
        'workers': cfg.num_workers,
        'fsdp_devices': cfg.fsdp_devices,
        'seed': cfg.seed,
        'action_dimensions': 32,
        'semantic_action_dimensions': 7,
        'wrench_start_end': [19, 25],
        'action_horizon': 50,
        'trainable_allow_regex': pilot.ALLOW,
        'optimizer': dataclasses.asdict(cfg.optimizer),
        'schedule': dataclasses.asdict(cfg.lr_schedule),
        'dataset_contract': dataclasses.asdict(contract),
        'source_sha256': {p.relative_to(WT).as_posix() if p.is_relative_to(WT) else str(p): sha(p) for p in files},
        'save_every': args.save_every,
    }
    return cfg, contract, manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('preflight', 'start', 'resume'), required=True)
    p.add_argument('--experiment', required=True)
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--workers', type=int, choices=(0, 4), default=4)
    p.add_argument('--save-every', type=int, default=2)
    p.add_argument('--stop-after', type=int, default=0)
    a = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_]+', a.experiment):
        p.error('Experiment must be an alphanumeric/underscore identifier')
    if not (a.steps == 100000 and 1 <= a.save_every <= a.steps):
        p.error('Invalid steps/save interval')
    if a.stop_after and not 1 <= a.stop_after <= a.steps:
        p.error('Invalid stop-after boundary')
    cfg, contract, manifest = prepare(a)
    run_dir = LOGROOT / a.experiment
    if a.mode == 'preflight':
        print('CONFIG_AND_SCIENCE_CONTRACT=PASS', flush=True)
        print(f'SAMPLE_BATCHES_PER_EPOCH={contract.batches_per_epoch}', flush=True)
        print(f'CHECKPOINT_DIR={cfg.checkpoint_dir}', flush=True)
        print('TRAINING_STARTED=NO', flush=True)
        return
    if jax.device_count() != 2 or any(d.platform != 'gpu' for d in jax.devices()):
        raise RuntimeError('Required exactly two workstation GPUs')
    if a.mode == 'start':
        if run_dir.exists() or cfg.checkpoint_dir.exists():
            raise FileExistsError('New run or checkpoint root already exists')
        run_dir.mkdir(parents=True)
        atomic_json(run_dir / 'scientific_contract.json', manifest, exclusive=True)
    else:
        if not (run_dir / 'scientific_contract.json').is_file() or not cfg.checkpoint_dir.is_dir():
            raise RuntimeError('Missing run manifest or existing checkpoint')
        if json.loads((run_dir / 'scientific_contract.json').read_text()) != manifest:
            raise RuntimeError('Scientific contract mismatch; resume blocked')
    with open(run_dir / 'run.lock', 'a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another ForceVLA writer owns the run') from exc
        journal = run_dir / 'telemetry.jsonl'
        invocation = f'{time.time_ns()}-{os.getpid()}'
        signal.signal(signal.SIGTERM, _request_stop)
        signal.signal(signal.SIGINT, _request_stop)
        base = load_source(BASE_TRAIN, 'forcevla_training_functions')
        rng = jax.random.key(cfg.seed)
        train_rng, init_rng = jax.random.split(rng)
        mesh = sharding.make_mesh(cfg.fsdp_devices)
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        manager, resuming = checkpoints.initialize_checkpoint_dir(
            cfg.checkpoint_dir, keep_period=None, overwrite=False,
            resume=(a.mode == 'resume'), max_to_keep=None,
        )
        if bool(resuming) != (a.mode == 'resume'):
            raise RuntimeError('Orbax checkpoint resume-state disagreement')
        try:
            expected_step = 0
            if a.mode == 'resume':
                receipt = json.loads((run_dir / 'latest_committed.json').read_text())
                expected_step = int(receipt['completed_updates'])
                validate_receipt(cfg.checkpoint_dir / str(expected_step), receipt)
                latest = max(map(int, manager.all_steps()))
                if latest != expected_step:
                    raise RuntimeError('Latest Orbax step and receipt differ')
            state, state_sharding = base.init_train_state(cfg, init_rng, mesh, resume=resuming)
            if resuming:
                state = checkpoints.restore_state(manager, state, None, step=expected_step)
            jax.block_until_ready(state)
            completed = int(np.asarray(jax.device_get(state.step)))
            if completed != expected_step:
                raise RuntimeError(f'Restored state.step={completed}, receipt={expected_step}')
            validate_cursor(contract, contract.cursor(completed), restored_step=completed)
            if completed >= cfg.num_train_steps:
                print(f'ALREADY_COMPLETED={completed}', flush=True)
                return
            loader = create_replay_data_loader(
                cfg, contract, completed_updates=completed,
                target_updates=cfg.num_train_steps,
                info_path=INFO, norm_stats_path=NORM,
                sharding=data_sharding,
            )
            it = iter(loader)
            step_fn = jax.jit(
                functools.partial(base.train_step, cfg),
                in_shardings=(replicated, state_sharding, data_sharding),
                out_shardings=(state_sharding, replicated),
                donate_argnums=(1,),
            )
            log_event(journal, event='resume' if resuming else 'start', invocation=invocation,
                      completed_updates=completed, checkpoint_step=expected_step)
            last_checkpoint = expected_step
            lr_fn = cfg.lr_schedule.create()
            for _ in range(completed, cfg.num_train_steps):
                # Batch is fetched only AFTER previous update's save/stop decisions.
                start = time.monotonic()
                batch = next(it)
                fetch_seconds = time.monotonic() - start
                previous = completed
                with sharding.set_mesh(mesh):
                    state, info = step_fn(train_rng, state, batch)
                jax.block_until_ready(state)
                values = jax.device_get(info)
                completed = int(np.asarray(jax.device_get(state.step)))
                if completed != previous + 1 or completed > cfg.num_train_steps:
                    raise RuntimeError(f'Optimizer-step discontinuity: {previous} -> {completed}')
                metrics = {k: float(np.asarray(v)) for k, v in values.items()}
                if not all(np.isfinite(v) for v in metrics.values()):
                    raise FloatingPointError(f'Nonfinite step metrics at {completed}')
                lr = float(np.asarray(jax.device_get(lr_fn(completed - 1))))
                log_event(journal, event='update', invocation=invocation,
                          completed_updates=completed, loss=metrics['loss'],
                          grad_norm=metrics.get('grad_norm'), param_norm=metrics.get('param_norm'),
                          learning_rate_used=lr, fetch_seconds=round(fetch_seconds, 5),
                          update_seconds=round(time.monotonic() - start - fetch_seconds, 5),
                          consumed_samples=completed * cfg.batch_size,
                          replay_cursor=contract.cursor(completed))
                print(f'UPDATE={completed}/{cfg.num_train_steps} LOSS={metrics["loss"]:.5f}', flush=True)
                requested = _STOP or (run_dir / 'stop.request').exists() or (
                    bool(a.stop_after) and completed >= a.stop_after
                )
                save = completed % cfg.save_interval == 0 or requested or completed == cfg.num_train_steps
                if save:
                    log_event(journal, event='checkpoint_begin', invocation=invocation, completed_updates=completed)
                    checkpoints.save_state(manager, state, loader, completed)
                    receipt = publish_committed(manager, cfg.checkpoint_dir, run_dir, completed)
                    last_checkpoint = completed
                    log_event(journal, event='checkpoint_checksummed', invocation=invocation,
                              completed_updates=completed, file_count=len(receipt['files']))
                    print(f'CHECKPOINT_COMMITTED={completed}', flush=True)
                if requested:
                    log_event(journal, event='safe_stop', invocation=invocation,
                              completed_updates=completed, last_checkpoint=last_checkpoint)
                    print(f'SAFE_STOP=PASS STEP={completed}', flush=True)
                    return
            log_event(journal, event='train_end', invocation=invocation,
                      completed_updates=completed, last_checkpoint=last_checkpoint)
            print(f'TRAIN_END=PASS STEP={completed}', flush=True)
        except Exception as exc:
            log_event(journal, event='fatal', invocation=invocation,
                      error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            manager.wait_until_finished()
            manager.close()


if __name__ == '__main__':
    main()
