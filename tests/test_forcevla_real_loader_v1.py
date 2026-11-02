"""CPU test against real ForceVLA LeRobot train split.

Tests:
- original transformation pipeline
- deterministic batch indices
- 0 versus 4 worker equality
- resume at a completed optimizer-step boundary
- 25D/32D/camera/language contract
- no training or model weight loading
"""

import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np

from openpi.training.forcevla_replay import (
    ReplayBatchSampler,
    ReplayContract,
)

from openpi.training.forcevla_reliable_loader import (
    create_replay_data_loader,
)

ROOT = Path(
    "/home/ktt_rc/robotics_thesis/forcevla_lab"
)

WT = ROOT / "worktrees/forcevla_lora_reliability_v1"
DATA = Path(
    "/mnt/robotics_ssd/ktt_rc_robotics_thesis/dataset"
)

INFO = (
    DATA
    / "forcevla_lerobot_language_v2_25d_train"
    / "meta/info.json"
)

NORM = (
    ROOT
    / "finetune_runs/assets"
    / "doosan_forcevla_25d_32d_compat_contract"
    / "forcevla_lerobot_language_v2_25d_train"
    / "norm_stats.json"
)


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def config_factory():
    source = (
        WT
        / "thesis_experiments"
        / "forcevla_lora_pilot_20261030"
        / "scripts/forcevla_lora_mock.py"
    )

    spec = importlib.util.spec_from_file_location(
        "forcevla_stage2_config",
        source,
    )

    assert spec and spec.loader

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.make_config(
        "REPLAY_STAGE2_NO_TRAINING", 8
    )


def digest_array(h, value):
    array = np.ascontiguousarray(
        np.asarray(jax.device_get(value))
    )

    h.update(str(array.shape).encode())
    h.update(str(array.dtype).encode())
    h.update(array.tobytes())


def digest_batch(batch):
    observation, actions = batch

    assert observation.state.shape == (2, 25)
    assert actions.shape == (2, 50, 32)

    assert set(observation.images) == {
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    }

    assert observation.tokenized_prompt.shape[0] == 2

    assert not np.any(
        np.asarray(
            observation.image_masks[
                "right_wrist_0_rgb"
            ]
        )
    )

    assert np.isfinite(
        np.asarray(actions)
    ).all()

    assert np.allclose(
        np.asarray(actions)[..., 7:],
        0.0,
    )

    h = hashlib.sha256()

    digest_array(h, observation.state)
    digest_array(h, actions)

    for name in sorted(observation.images):
        digest_array(
            h, observation.images[name]
        )
        digest_array(
            h, observation.image_masks[name]
        )

    digest_array(h, observation.tokenized_prompt)
    digest_array(
        h, observation.tokenized_prompt_mask
    )

    return h.hexdigest()


def main(report_path):
    assert len(jax.devices()) == 1
    assert jax.devices()[0].platform == "cpu"

    info = json.loads(INFO.read_text())

    assert info["total_episodes"] == 255
    assert info["total_frames"] == 289703

    contract = ReplayContract(
        dataset_size=info["total_frames"],
        batch_size=2,
        seed=42,
        dataset_info_sha256=sha(INFO),
        norm_stats_sha256=sha(NORM),
    )

    base = config_factory()

    assert base.model.state_dim == 25
    assert base.model.action_dim == 32
    assert base.model.action_horizon == 50

    def collect(start, end, workers):
        cfg = replace(
            base,
            num_workers=workers,
            fsdp_devices=1,
        )

        loader = create_replay_data_loader(
            cfg,
            contract,
            completed_updates=start,
            target_updates=end,
            info_path=INFO,
            norm_stats_path=NORM,
        )

        return [
            digest_batch(batch)
            for batch in loader
        ]

    print("TEST_REAL_LOADER_WORKERS_0", flush=True)
    original = collect(0, 2, 0)

    assert len(original) == 2

    print("TEST_REAL_LOADER_RESUME", flush=True)
    resumed = collect(1, 2, 0)

    assert resumed == original[1:]

    print("TEST_REAL_LOADER_WORKERS_4", flush=True)
    workers4 = collect(0, 2, 4)

    assert workers4 == original

    sampler = ReplayBatchSampler(
        contract,
        completed_updates=0,
        target_updates=2,
    )

    indices = list(sampler)

    assert len(indices) == 2
    assert all(len(batch) == 2 for batch in indices)

    result = {
        "schema": "forcevla-real-loader-integration-v1",
        "training_performed": False,
        "dataset_frames": info["total_frames"],
        "batch_size": contract.batch_size,
        "workers_tested": [0, 4],
        "replay_start": 1,
        "test_batches": 2,
        "batch_indices": indices,
        "batch_sha256": original,
        "resume_batch_equality": True,
        "worker_batch_equality": True,
        "dataset_contract_sha256": contract.digest,
        "model_contract": {
            "state_dim": 25,
            "action_dim": 32,
            "action_horizon": 50,
        },
        "result": "PASS",
    }

    path = Path(report_path)

    with path.open("x") as stream:
        json.dump(result, stream, indent=2)

    print("REAL_TRANSFORMS=PASS")
    print("REAL_RESUME_BATCH_EQUALITY=PASS")
    print("WORKERS_0_VS_4_EQUALITY=PASS")
    print("MODEL_INPUT_SHAPES=PASS")
    print("PADDED_ACTIONS_ZERO=PASS")
    print("RESULT_FILE=", path)
    print("STAGE_2=PASS")


if __name__ == "__main__":
    assert len(sys.argv) == 2
    main(sys.argv[1])
