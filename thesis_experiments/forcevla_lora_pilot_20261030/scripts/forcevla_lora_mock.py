import importlib.util
import json
import math
import re
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import flax.nnx as nnx
import jax

from openpi.shared import nnx_utils
from openpi.training import config as training_config
from openpi.training import weight_loaders

ROOT = Path("/home/ktt_rc/robotics_thesis/forcevla_lab")
DATA_PARENT = Path(
    "/mnt/robotics_ssd/ktt_rc_robotics_thesis/dataset"
)
NAME = "forcevla_lerobot_language_v2_25d_train"
CONTRACT = "doosan_forcevla_25d_32d_compat_contract"

# Explicit trainable allowlist:
#  - LoRA adapters in Gemma language/action experts
#  - force fusion and LIMoE
#  - proprioception and action projections
# SigLIP and original Gemma weights are frozen.
ALLOW = (
    r"(?:PaliGemma/llm/.*lora.*"
    r"|force_in_proj/.*"
    r"|limoe/.*"
    r"|state_proj/.*"
    r"|action_in_proj/.*"
    r"|action_out_proj/.*"
    r"|action_time_mlp_in/.*"
    r"|action_time_mlp_out/.*)"
)
PATTERN = re.compile(ALLOW)

def make_config(experiment, steps):
    base = training_config.get_config(CONTRACT)
    return replace(
        base,
        exp_name=experiment,
        project_name="doosan-forcevla-lora-mock",
        data=replace(base.data, repo_id=NAME),
        assets_base_dir=str(ROOT / "finetune_runs/assets"),
        checkpoint_base_dir=str(
            DATA_PARENT / "forcevla_mock_runs/checkpoints"
        ),
        weight_loader=weight_loaders.Pi0GuidanceWeightLoader(
            str(ROOT / "openpi_checkpoints/pi0_base/params")
        ),
        freeze_filter=nnx.Not(nnx_utils.PathRegex(ALLOW)),
        num_train_steps=steps,
        batch_size=2,
        num_workers=0,
        fsdp_devices=2,
        ema_decay=None,
        log_interval=1,
        save_interval=100,
        keep_period=None,
        wandb_enabled=False,
        overwrite=False,
        resume=False,
    )

def preflight(cfg, audit=True):
    train = DATA_PARENT / NAME
    info = json.loads((train / "meta/info.json").read_text())

    assert (train / ".train_view_complete").is_file()
    assert info["total_episodes"] == 255
    assert info["total_frames"] == 289703
    assert info["splits"] == {"train": "0:255"}

    assert (ROOT / "openpi_checkpoints/pi0_base/params").is_dir()
    assert not cfg.checkpoint_dir.exists()

    assert cfg.model.state_dim == 25
    assert cfg.model.action_dim == 32
    assert cfg.model.action_horizon == 50
    assert cfg.fsdp_devices == 2

    stats = cfg.data.create(cfg.assets_dirs, cfg.model).norm_stats
    assert stats is not None
    assert stats["state"].mean.shape == (25,)
    assert stats["actions"].mean.shape == (32,)

    # Reserve space for the full-state Orbax checkpoint.
    free_gib = shutil.disk_usage(DATA_PARENT).free / 2**30
    assert free_gib >= 20, (
        f"Insufficient SSD free space: {free_gib:.1f} GiB"
    )

    print("DATASET_AND_SPLITS=PASS", flush=True)
    print("NORMALIZATION=PASS", flush=True)
    print(f"SSD_FREE_GIB={free_gib:.1f}", flush=True)

    if audit:
        # Abstract model construction: no pretrained weight loading.
        state = jax.eval_shape(
            lambda: nnx.state(cfg.model.create(jax.random.key(0)))
        )

        all_params = state.filter(nnx.Param).flat_state()
        selected = state.filter(cfg.trainable_filter).flat_state()

        paths = [
            "/".join(map(str, path))
            for path in selected.keys()
        ]

        assert paths and len(paths) < len(all_params)
        assert all(PATTERN.fullmatch(path) for path in paths)

        checks = {
            "LORA": any(
                path.startswith("PaliGemma/llm/")
                and "lora" in path for path in paths
            ),
            "FORCE_PROJECTION": any(
                path.startswith("force_in_proj/") for path in paths
            ),
            "LIMOE": any(
                path.startswith("limoe/") for path in paths
            ),
            "STATE_PROJECTION": any(
                path.startswith("state_proj/") for path in paths
            ),
            "ACTION_PROJECTION": any(
                path.startswith("action_") for path in paths
            ),
            "VISION_FROZEN": not any(
                path.startswith("PaliGemma/img/")
                for path in paths
            ),
            "BASE_GEMMA_FROZEN": not any(
                path.startswith("PaliGemma/llm/")
                and "lora" not in path
                for path in paths
            ),
        }

        for key, passed in checks.items():
            print(
                f"{key}={'PASS' if passed else 'FAIL'}",
                flush=True,
            )
        assert all(checks.values())

        def elements(variables):
            return sum(
                math.prod(var.value.shape)
                for var in variables.values()
            )

        total = elements(all_params)
        trainable = elements(selected)
        print(f"PARAMETERS_TOTAL={total}", flush=True)
        print(f"PARAMETERS_TRAINABLE={trainable}", flush=True)
        print(
            f"TRAINABLE_PERCENT={100 * trainable / total:.3f}",
            flush=True,
        )

    print("FSDP_GPUS=2", flush=True)
    print(f"STEPS={cfg.num_train_steps}", flush=True)
    print(f"CHECKPOINT_DIR={cfg.checkpoint_dir}", flush=True)
    print("PREFLIGHT=PASS", flush=True)

def run(cfg):
    assert len(jax.devices()) == 2
    assert all(device.platform == "gpu" for device in jax.devices())

    path = ROOT / "ForceVLA_thesis/scripts/train.py"
    spec = importlib.util.spec_from_file_location(
        "forcevla_lora_train", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    print("TRAINING=STARTED", flush=True)
    module.main(cfg)
    print("TRAINING=COMPLETED", flush=True)

if __name__ == "__main__":
    assert len(sys.argv) == 4
    mode, experiment, raw_steps = sys.argv[1:]

    assert mode in ("preflight", "run")
    assert re.fullmatch(r"[A-Za-z0-9_]+", experiment)

    steps = int(raw_steps)
    assert 1 <= steps <= 30, "This launcher is limited to short mocks"

    cfg = make_config(experiment, steps)
    preflight(cfg, audit=(mode == "preflight"))

    if mode == "run":
        run(cfg)
