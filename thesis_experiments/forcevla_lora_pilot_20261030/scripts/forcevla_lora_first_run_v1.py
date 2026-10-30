import dataclasses
import hashlib
import importlib.util
import json
import re
import shutil
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path("/home/ktt_rc/robotics_thesis/forcevla_lab")
DATA = Path("/mnt/robotics_ssd/ktt_rc_robotics_thesis/dataset")
LOGDIR = ROOT / "logs/finetune_readiness"

TRAIN = DATA / "forcevla_lerobot_language_v2_25d_train"
STATS = (
    ROOT / "finetune_runs/assets/"
    "doosan_forcevla_25d_32d_compat_contract/"
    "forcevla_lerobot_language_v2_25d_train/norm_stats.json"
)
PILOT = ROOT / "scripts/forcevla_lora_mock.py"

SOURCE_FILES = [
    Path(__file__),
    PILOT,
    ROOT / "ForceVLA_thesis/scripts/train.py",
    ROOT / "ForceVLA_thesis/src/openpi/models/pi0_force.py",
    ROOT / "ForceVLA_thesis/src/openpi/training/checkpoints.py",
    TRAIN / "meta/info.json",
    STATS,
]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    assert len(sys.argv) == 3
    mode, exp = sys.argv[1:]
    assert mode in ("prepare", "start", "resume")
    assert re.fullmatch(r"[A-Za-z0-9_]+", exp)

    spec = importlib.util.spec_from_file_location(
        "fvla_lora_config", PILOT
    )
    assert spec and spec.loader

    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)

    cfg = replace(
        pilot.make_config(exp, 2000),
        fsdp_devices=1,
        batch_size=2,
        num_workers=4,
        log_interval=25,
        save_interval=500,
        keep_period=None,
        overwrite=False,
        resume=(mode == "resume"),
    )

    assert cfg.lr_schedule.warmup_steps == 1000
    assert cfg.num_train_steps == 2000
    assert cfg.model.state_dim == 25
    assert cfg.model.action_dim == 32
    assert cfg.model.action_horizon == 50
    assert cfg.data.repo_id == TRAIN.name
    assert cfg.ema_decay is None

    info = json.loads((TRAIN / "meta/info.json").read_text())

    assert info["total_episodes"] == 255
    assert info["total_frames"] == 289703
    assert info["splits"] == {"train": "0:255"}
    assert (TRAIN / ".train_view_complete").is_file()

    norm = cfg.data.create(
        cfg.assets_dirs, cfg.model
    ).norm_stats

    assert norm is not None
    assert norm["state"].mean.shape == (25,)
    assert norm["actions"].mean.shape == (32,)

    wrench_std = [
        float(v) for v in norm["state"].std[19:25]
    ]
    assert len(wrench_std) == 6
    assert all(v >= 0 for v in wrench_std)

    assert all(path.is_file() for path in SOURCE_FILES)

    fingerprints = {
        str(path): digest(path)
        for path in SOURCE_FILES
    }

    plan_path = LOGDIR / f"{exp}.plan.json"
    ckpt_dir = cfg.checkpoint_dir

    free_gib = shutil.disk_usage(DATA).free / 2**30

    print(f"DISK_FREE_GIB={free_gib:.1f}", flush=True)
    print(f"WRENCH_STDS={wrench_std}", flush=True)
    print("TRAIN_ONLY_DATA=PASS", flush=True)
    print("CONFIG=DP2_BATCH2_WORKERS4_2000STEPS", flush=True)
    print("CHECKPOINT_INTERVAL=500", flush=True)

    if mode == "prepare":
        assert not ckpt_dir.exists()
        assert free_gib >= 25, (
            "At least 25 GiB free space required"
        )

        plan = {
            "experiment": exp,
            "steps": 2000,
            "batch_size": 2,
            "fsdp_devices": 1,
            "num_workers": 4,
            "save_interval": 500,
            "log_interval": 25,
            "seed": cfg.seed,
            "optimizer": dataclasses.asdict(cfg.optimizer),
            "lr_schedule": dataclasses.asdict(cfg.lr_schedule),
            "checkpoint_dir": str(ckpt_dir),
            "dataset_dir": str(TRAIN),
            "freeze_allow_regex": pilot.ALLOW,
            "fingerprints": fingerprints,
            "resume_note": (
                "Weights, optimizer state and step resume; "
                "exact shuffled loader position does not."
            ),
        }

        with plan_path.open("x") as f:
            json.dump(plan, f, indent=2)

        print("PLAN=PASS", flush=True)
        print(f"MANIFEST={plan_path}", flush=True)
        return

    plan = json.loads(plan_path.read_text())

    assert plan["fingerprints"] == fingerprints, (
        "Source or dataset metadata changed"
    )
    assert plan["checkpoint_dir"] == str(ckpt_dir)
    assert plan["freeze_allow_regex"] == pilot.ALLOW

    if mode == "start":
        assert not ckpt_dir.exists(), (
            "Experiment exists; use resume instead"
        )
        assert free_gib >= 25
    else:
        assert ckpt_dir.is_dir()

        completed = sorted(
            int(p.name)
            for p in ckpt_dir.iterdir()
            if p.is_dir()
            and p.name.isdigit()
            and (p / "_CHECKPOINT_METADATA").is_file()
        )

        assert completed, "No finalized checkpoint available"

        print(
            f"RESUME_FROM_SAVED_STEP={completed[-1]}",
            flush=True,
        )

        if completed[-1] >= 1999:
            print(
                "ALREADY_FINISHED_2000_STEPS=PASS",
                flush=True,
            )
            return

        assert free_gib >= 18

    import jax

    devices = jax.devices()
    assert len(devices) == 2
    assert all(d.platform == "gpu" for d in devices)

    print(
        f"TRAINING_MODE={mode.upper()}",
        flush=True,
    )

    pilot.run(cfg)
    print("TRAINING_COMPLETED=PASS", flush=True)

if __name__ == "__main__":
    main()
