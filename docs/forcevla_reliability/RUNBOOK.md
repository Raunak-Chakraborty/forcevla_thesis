# ForceVLA LoRA workstation runbook — reliability v1

**System:** lab workstation `ktt_rc@Precision-7920-Tower`, 2× RTX A5000, JAX/Flax NNX/Optax/Orbax and Torch/LeRobot. **No XR-1/CLAIX modifications.**

## Scope and qualification status

The October 30 2,000-step LoRA pilot and its historical checkpoint remain untouched. The new ForceVLA reliability worktree uses the same model, force/LIMoE/LoRA trainability, loss (including 32D action padding), dataset, normalizations and optimization formula. No last-N-layer fine-tuning was added.

Evidence: Stage 1 synthetic multi-epoch batch replay with 0/4 workers passed; Stage 2 actual LeRobot transformed-batch equality and fresh-cursor restart passed; Stage 3 CPU receipt corruption and journal tests passed; Stage 4 four-update stop/resume and uninterrupted control on 2× A5000 both trained to completion. All three resulting 5.77–5.81 GiB Orbax payloads were checksummed and receipts/pointers validated. Stage 4 strict numerical equality **did not pass**; discrepancies begin at update 1, before resume, with max absolute loss difference 0.01032543. Stage 4B real-input hashes matched across independent CPU processes. Exact optimizer-tensor equality and full independent checkpoint restore remain unverified. Do not claim bitwise deterministic model training.

## Operations (use lab workstation terminal)

Root: `/home/ktt_rc/robotics_thesis/forcevla_lab`

Controller: `worktrees/forcevla_lora_reliability_v1/scripts/forcevla_operator_v1.py`

Interpreter: `$HOME/miniforge3/envs/forcevla_lab_tf_test/bin/python`

Choose an **unused** experiment identifier with letters/digits/underscore. Example `LORA_FULL_REVIEWED_01` is only a placeholder. `--steps 30000 --save-every 1000 --workers 4` is a **proposal**, not a scientific approval.

- Plan (safe, no GPU): `python scripts/forcevla_operator_v1.py plan --experiment LORA_FULL_REVIEWED_01 --steps 30000 --save-every 1000`
- Status (safe, no GPU): `python scripts/forcevla_operator_v1.py status --experiment LORA_FULL_REVIEWED_01 --steps 30000`
- Start (GPU training, **only after explicit approval**): `python scripts/forcevla_operator_v1.py start --experiment LORA_FULL_REVIEWED_01 --steps 30000 --save-every 1000 --workers 4 --execute`
- Request safe stop (run must exist and be supervised): `python scripts/forcevla_operator_v1.py stop --experiment LORA_FULL_REVIEWED_01`
- Resume (GPU training, requires verified latest receipt and matching settings): `python scripts/forcevla_operator_v1.py resume --experiment LORA_FULL_REVIEWED_01 --steps 30000 --save-every 1000 --workers 4 --execute`

Run commands from the operator's absolute path or `cd` to the reliability worktree first. `plan` and `status` do not import the JAX trainer. The controller launches a detached supervisor and child with redirected stdout, a per-invocation record and independent GPU sampling. It prints paths, not promises that the child has started successfully. Inspect `status`, `trainer-*.log`, `supervisor-*.log` and `exit-*.json` after launch. Browser auto-open is never used.

## Path layout

- Per-run telemetry and receipts: `logs/lora_reliability/runs/<experiment>/telemetry.jsonl`, `scientific_contract.json`, `receipts/`, `latest_committed.json`
- Operator records: `logs/lora_reliability/operator/<experiment>/invocation-*.json`, `latest_launch.json`, `worker_state.json`, `trainer-*.log`, `gpu-*.jsonl`, `exit-*.json`
- Orbax payloads: `/mnt/robotics_ssd/ktt_rc_robotics_thesis/dataset/forcevla_reliable_runs/checkpoints/doosan_forcevla_25d_32d_compat_contract/<experiment>/<completed_updates>/`

The checkpoint folder label is **completed optimizer updates**, unlike the historical pilot's `1999` directory. The training loader cursor is defined by `TrainState.step`, which increments **only after completing an optimizer update**. The control step is safe after the update and before fetching the next batch. The trainer's own `flock` prevents simultaneous writes to the *same run*, while the operator separately prevents starting another supervised experiment when it detects an active trainer. No checkpoint is overwritten or pruned by this release.

The `latest_committed.json` indicates payload bytes were hashed and Orbax finalized at the time of publication; it is **not** proof of independent full-state restore. The resume path rehashes the payload, restores the full Orbax state and validates the restored step. The operator's status command intentionally does **not** reread ~5.8 GiB to rehash at every status query.

## Storage and schedule risk

With 5.85 GiB per checkpoint and 30,000 updates saving every 1,000, expect ~30 regular saves, plus safe-stop saves, metadata and transient headroom. The operator computes a conservative capacity budget, but no automatic pruning is implemented: disk should be monitored. Batch size 2 × 30,000 updates = 60,000 samples, **roughly 20.7%** of the 289,703-frame training split, not a complete epoch. Longer horizons exceed the current trainer's `--steps <= 30000` guard and require an explicit reviewed change in scheduler/experiment contract. Do not silently extend an existing run.

A single long run may take many hours and its actual rate must be measured from the first sustained updates. The controller ETA is provisional and based on recent telemetry, not a guaranteed wallclock deadline. Ensure the workstation stays powered on, storage remains mounted, and the GPUs are not needed by another user. Stop requests are only checked at completed optimizer boundaries: if the data loader is blocked before the next update, the stop will not be immediate.

## Failure and re-run safety

A launcher exit means the supervisor's *launch request* completed, not that GPU training succeeded. Always inspect `status` and the exit receipt. A crash may leave an append-only failed telemetry tail; retain it, and use only committed checkpoint receipts for authoritative restart points. Do not remove `latest_committed.json`, modify `scientific_contract.json`, overwrite Orbax checkpoints, or edit the reliability worktree between start and resume. The operator archives a pending `stop.request` when launching a reviewed resume; do not manually delete it during an active run. Always save the final Git commit hash and the accepted scientific plan prior to an approved full run.
