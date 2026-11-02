#!/usr/bin/env python3
"""Workstation-only ForceVLA LoRA operator. Pure host Python; no JAX import.

Launch is opt-in (--execute); 'plan' and 'status' cannot start training.
For training internals and scientific fingerprints see train_forcevla_reliable_v1.py.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path('/home/ktt_rc/robotics_thesis/forcevla_lab')
WT = ROOT / 'worktrees/forcevla_lora_reliability_v1'
DATA = Path('/mnt/robotics_ssd/ktt_rc_robotics_thesis/dataset')
PY = Path('/home/ktt_rc/miniforge3/envs/forcevla_lab_tf_test/bin/python')
LOGROOT = ROOT / 'logs/lora_reliability'
RUNROOT = LOGROOT / 'runs'
OPSROOT = LOGROOT / 'operator'
CKROOT = DATA / 'forcevla_reliable_runs/checkpoints/doosan_forcevla_25d_32d_compat_contract'
TRAINER = WT / 'scripts/train_forcevla_reliable_v1.py'
REMOTE = 'https://github.com/Raunak-Chakraborty/forcevla_thesis.git'
BRANCH = 'thesis/forcevla-lora-reliability-v1'


def utc():
    return datetime.now(timezone.utc).isoformat()


def utc_stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def atomic_json(path: Path, obj):
    if path.is_symlink():
        raise RuntimeError('Refusing symlink output: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(obj, f, sort_keys=True, indent=2)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def read_json(path):
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f'Missing/unsafe JSON file: {path}')
    return json.loads(path.read_text())


def run_cmd(args, *, timeout=12):
    p = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(f'Command failed {p.returncode}: {args!r}\n{p.stderr[-500:]}')
    return p.stdout.strip()


def host_preflight():
    if not PY.is_file() or not TRAINER.is_file():
        raise RuntimeError('Missing ForceVLA interpreter or reliable trainer')
    if run_cmd(['git', '-C', str(WT), 'branch', '--show-current']) != BRANCH:
        raise RuntimeError('Wrong reliability branch')
    if run_cmd(['git', '-C', str(WT), 'status', '--porcelain']):
        raise RuntimeError('Reliability Git worktree is dirty; commit/review changes first')
    if run_cmd(['git', '-C', str(WT), 'remote', 'get-url', 'thesis_github']) != REMOTE:
        raise RuntimeError('Unexpected GitHub remote')
    if not (DATA / 'forcevla_lerobot_language_v2_25d_train/meta/info.json').is_file():
        raise RuntimeError('Training dataset metadata missing')
    return run_cmd(['git', '-C', str(WT), 'rev-parse', 'HEAD'])


def free_gib():
    st = os.statvfs(DATA)
    return st.f_bavail * st.f_frsize / 2**30


def requirements(steps, every):
    saves = math.ceil(steps / every) + 2  # possible stop and transient write
    expected_gib = saves * 5.85 + 64.0  # conservative budget including headroom
    return saves, expected_gib


def proc_start_ticks(pid):
    try:
        # proc stat field 22: starttime; fields after closing ')' start at field 3.
        raw = Path(f'/proc/{pid}/stat').read_text()
        return int(raw.rsplit(')', 1)[1].strip().split()[19])
    except (FileNotFoundError, ProcessLookupError, ValueError, IndexError, PermissionError):
        return None


def process_alive(record):
    try:
        return (int(record['pid']) > 0 and proc_start_ticks(int(record['pid']))
                == int(record['start_ticks']))
    except (KeyError, TypeError, ValueError):
        return False


def current_launch(ops):
    path = ops / 'latest_launch.json'
    return read_json(path) if path.is_file() else None


def current_worker(ops):
    path = ops / 'worker_state.json'
    return read_json(path) if path.is_file() else None


def active(ops):
    launch = current_launch(ops)
    worker = current_worker(ops)
    return bool((launch and process_alive(launch)) or
                (worker and worker.get('state') == 'running' and process_alive(worker)))


def external_trainers():
    others = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / 'cmdline').read_bytes().split(b'\0')
        except (OSError, PermissionError):
            continue
        # Match argv components, not script text in a parent shell heredoc.
        if any(arg.endswith(b'/train_forcevla_reliable_v1.py') or
               arg.endswith(b'/forcevla_lora_first_run_v1.py') for arg in argv):
            others.append(int(entry.name))
    return sorted(others)


def gpu_snapshot():
    try:
        output = run_cmd(['nvidia-smi', '--query-gpu=index,uuid,memory.free,memory.total,utilization.gpu',
                          '--format=csv,noheader,nounits'], timeout=8)
        return output.splitlines()
    except Exception as e:
        return ['UNAVAILABLE: ' + str(e)]


def telemetry_recent(run_dir):
    p = run_dir / 'telemetry.jsonl'
    if not p.is_file():
        return [], []
    with p.open('r', encoding='utf-8') as f:
        lines = deque(f, maxlen=700)
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows, [r for r in rows if r.get('event') == 'update']


def status(experiment, steps):
    run_dir, ops = RUNROOT / experiment, OPSROOT / experiment
    launched = current_launch(ops)
    worker = current_worker(ops)
    running = active(ops)
    rows, updates = telemetry_recent(run_dir)
    last_step = max((int(r['completed_updates']) for r in updates), default=0)
    print(f'EXPERIMENT={experiment}')
    print(f'LAUNCHER_LIVE={running}')
    if launched:
        print(f'LAST_LAUNCH_PID={launched.get("pid")}')
        print(f'LAST_LAUNCH_ID={launched.get("invocation_id")}')
    if worker:
        print(f'WORKER_STATE={worker.get("state")}')
        print(f'TRAINER_PID={worker.get("trainer_pid", "UNKNOWN")}')
        print(f'TRAINER_LOG={worker.get("trainer_log", "UNKNOWN")}')
    print(f'LAST_LOGGED_COMPLETED_UPDATES={last_step}')
    latest = run_dir / 'latest_committed.json'
    if latest.is_file():
        rec = read_json(latest)
        print(f'LAST_CHECKSUMMED_CHECKPOINT_STEP={rec.get("completed_updates")}')
        print('CHECKPOINT_NOTE=Checksummed at publication; status does not rehash payload')
    else:
        print('LAST_CHECKSUMMED_CHECKPOINT_STEP=NONE')
    if updates:
        print(f'LAST_LOSS={updates[-1].get("loss")}')
        print(f'LAST_LR_USED={updates[-1].get("learning_rate_used")}')
        print(f'LAST_UPDATE_UTC={updates[-1].get("utc")}')
        last_inv = updates[-1].get('invocation')
        sequence = [r for r in updates if r.get('invocation') == last_inv]
        if len(sequence) >= 2:
            dt = (datetime.fromisoformat(sequence[-1]['utc']) -
                  datetime.fromisoformat(sequence[0]['utc'])).total_seconds()
            n = sequence[-1]['completed_updates'] - sequence[0]['completed_updates']
            if dt > 0 and n > 0:
                sec_per = dt / n
                print(f'MEASURED_SECONDS_PER_UPDATE_IN_INVOCATION={sec_per:.2f}')
                print(f'OPTIMISTIC_ETA_HOURS={(max(steps-last_step, 0) * sec_per / 3600):.2f}')
                print('ETA_NOTE=Recent interval only; future checkpoint/IO variability excluded')
    if rows:
        print('LAST_TELEMETRY_EVENT=' + str(rows[-1].get('event')))
    if (run_dir / 'stop.request').exists():
        print('STOP_REQUEST=PENDING')
    exit_records = sorted(ops.glob('exit-*.json')) if ops.is_dir() else []
    if exit_records:
        print('LAST_SUPERVISOR_EXIT=' + str(read_json(exit_records[-1]).get('exit_code')))
    print('GPU_STATUS=' + ' | '.join(gpu_snapshot())[:500])


def gpu_monitor(destination: Path, stopping: threading.Event):
    while not stopping.is_set():
        try:
            lines = gpu_snapshot()
            with destination.open('a') as f:
                f.write(json.dumps({'utc': utc(), 'gpu': lines}) + '\n')
        except Exception:
            pass  # telemetry monitor must not interrupt training
        stopping.wait(30)


def worker_main(record_path):
    record = read_json(record_path)
    experiment = record['experiment']
    ops = OPSROOT / experiment
    invocation = record['invocation_id']
    log = ops / f'trainer-{invocation}.log'
    exit_path = ops / f'exit-{invocation}.json'
    env = os.environ.copy()
    env['PYTHONPATH'] = ':'.join([
        str(WT / 'src'), str(ROOT / 'ForceVLA_thesis/lerobot'),
        str(ROOT / 'ForceVLA_thesis/dlimp')])
    env['HF_LEROBOT_HOME'] = str(DATA)
    env['HF_DATASETS_CACHE'] = str(DATA / '.forcevla_hf_datasets_cache')
    env['HF_HUB_OFFLINE'] = env['HF_DATASETS_OFFLINE'] = env['TRANSFORMERS_OFFLINE'] = '1'
    env['CUDA_VISIBLE_DEVICES'] = '0,1'
    env.pop('JAX_PLATFORMS', None)
    env.update({'XLA_PYTHON_CLIENT_PREALLOCATE': 'false', 'TOKENIZERS_PARALLELISM': 'false',
                'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                'OPENBLAS_NUM_THREADS': '2'})
    cmd = [str(PY), '-u', str(TRAINER), '--mode', record['mode'], '--experiment', experiment,
           '--steps', str(record['steps']), '--workers', str(record['workers']),
           '--save-every', str(record['save_every'])]
    monitor_stop = threading.Event()
    monitor = threading.Thread(target=gpu_monitor, args=(ops / f'gpu-{invocation}.jsonl', monitor_stop), daemon=True)
    rc = 99
    child_pid = None
    try:
        with log.open('x') as output:
            child = subprocess.Popen(cmd, cwd=WT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=output, stderr=subprocess.STDOUT)
            child_pid = child.pid
            atomic_json(ops / 'worker_state.json', {'state': 'running', 'invocation_id': invocation,
                'trainer_pid': child_pid, 'trainer_start_ticks': proc_start_ticks(child_pid),
                'trainer_log': str(log), 'utc': utc()})
            try:
                monitor.start()
            except Exception as error:
                with (ops / f'monitor-error-{invocation}.json').open('x') as f:
                    json.dump({'utc': utc(), 'error': repr(error)}, f)
            rc = child.wait()
    except Exception as error:
        atomic_json(ops / f'worker-error-{invocation}.json', {'error': repr(error), 'utc': utc()})
        raise
    finally:
        monitor_stop.set()
        if monitor.is_alive():
            monitor.join(timeout=5)
        atomic_json(exit_path, {'utc': utc(), 'invocation_id': invocation,
                                'trainer_pid': child_pid, 'exit_code': rc, 'trainer_log': str(log)})
        atomic_json(ops / 'worker_state.json', {'state': 'exited', 'invocation_id': invocation,
            'exit_code': rc, 'trainer_log': str(log), 'utc': utc()})
    return rc


def submit(args):
    commit = host_preflight()
    ops = OPSROOT / args.experiment
    ops.mkdir(parents=True, exist_ok=True)
    lock_path = ops / 'operator.lock'
    # Serialize launches across *all* experiments, not just this run.
    global_lock_path = OPSROOT / '.global_launch.lock'
    with global_lock_path.open('a+') as global_lock, lock_path.open('a+') as lock:
        fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(active(item) for item in OPSROOT.iterdir() if item.is_dir()):
            raise RuntimeError('An existing ForceVLA supervisor is active')
        if external_trainers():
            raise RuntimeError('Another ForceVLA trainer is active: ' + repr(external_trainers()))
        if not (1 <= args.save_every <= args.steps <= 30000):
            raise RuntimeError('Invalid checkpoint interval or training horizon')
        n, budget = requirements(args.steps, args.save_every)
        if free_gib() < budget:
            raise RuntimeError(f'Insufficient free SSD: {free_gib():.1f} GiB < {budget:.1f} GiB required')
        gpus = gpu_snapshot()
        if len(gpus) < 2 or any('UNAVAILABLE' in line for line in gpus):
            raise RuntimeError('Both GPUs required, nvidia-smi unavailable')
        import csv
        for row in csv.reader(gpus[:2], skipinitialspace=True):
            if float(row[2]) < 14000:
                raise RuntimeError('At least 14 GiB free GPU RAM required on each A5000')
        run_dir, ckdir = RUNROOT / args.experiment, CKROOT / args.experiment
        if args.action == 'start':
            if run_dir.exists() or ckdir.exists():
                raise RuntimeError('New experiment exists; use resume, never overwrite')
        else:
            latest = read_json(run_dir / 'latest_committed.json')
            done = int(latest['completed_updates'])
            if done >= args.steps:
                raise RuntimeError(f'Experiment already finished at {done} updates')
            existing = read_json(run_dir / 'scientific_contract.json')
            if (existing.get('target_updates') != args.steps or
                existing.get('workers') != args.workers or
                existing.get('save_every') != args.save_every):
                raise RuntimeError('Resume settings differ from recorded run contract')
        invocation = utc_stamp() + '-' + str(os.getpid())
        record = {'schema': 'forcevla-workstation-operator-v1', 'experiment': args.experiment,
                  'steps': args.steps, 'workers': args.workers, 'save_every': args.save_every,
                  'mode': args.action, 'invocation_id': invocation, 'utc': utc(),
                  'git_commit': commit, 'checkpoint_budget_gib': round(budget, 2)}
        path = ops / f'invocation-{invocation}.json'
        atomic_json(path, record)
        if args.action == 'resume':
            stop = run_dir / 'stop.request'
            if stop.exists():
                stop.rename(run_dir / f'stop.request.ack-{invocation}.json')
        sup_log = ops / f'supervisor-{invocation}.log'
        with sup_log.open('x') as output:
            proc = subprocess.Popen([str(PY), '-u', str(Path(__file__).resolve()),
                                     '--internal-worker', str(path)], cwd=WT,
                                     stdin=subprocess.DEVNULL, stdout=output,
                                     stderr=subprocess.STDOUT, start_new_session=True,
                                     close_fds=True)
        atomic_json(ops / 'latest_launch.json', {'pid': proc.pid, 'start_ticks': proc_start_ticks(proc.pid),
            'invocation_id': invocation, 'supervisor_log': str(sup_log), 'utc': utc()})
        print('TRAINING_LAUNCH_REQUESTED=YES')
        print('SUPERVISOR_PID=' + str(proc.pid))
        print('INVOCATION_RECORD=' + str(path))
        print('SUPERVISOR_LOG=' + str(sup_log))
        print('STATUS_COMMAND=' + str(PY) + ' ' + str(Path(__file__)) +
              f' status --experiment {args.experiment} --steps {args.steps}')
        print('LAUNCH_NOTE=Check the status/log; process startup is not yet verified')


def stop(experiment):
    run_dir = RUNROOT / experiment
    if not run_dir.is_dir():
        raise RuntimeError('Run directory does not exist; stop request would corrupt future start')
    if not active(OPSROOT / experiment):
        raise RuntimeError('No live supervised ForceVLA run; inspect status first')
    path = run_dir / 'stop.request'
    with path.open('x') as f:
        json.dump({'utc': utc(), 'reason': 'operator_request'}, f)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    print('SAFE_STOP_REQUESTED=YES')
    print('STOP_REQUEST=' + str(path))
    print('NOTE=Trainer stops at next completed optimizer boundary and finishes Orbax publication')


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--internal-worker':
        return worker_main(Path(sys.argv[2]))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['plan', 'start', 'resume', 'status', 'stop'])
    p.add_argument('--experiment', required=True)
    p.add_argument('--steps', type=int, default=30000)
    p.add_argument('--workers', type=int, choices=[0, 4], default=4)
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--execute', action='store_true', help='Explicitly permit detached GPU training')
    a = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_]+', a.experiment):
        p.error('Experiment must contain only letters, digits and underscore')
    if a.action == 'plan':
        if not 2 <= a.steps <= 30000 or not 1 <= a.save_every <= a.steps:
            p.error('Invalid steps or save interval')
        saves, needed = requirements(a.steps, a.save_every)
        print('MODE=PLAN_ONLY')
        print(f'EXPERIMENT={a.experiment}')
        print(f'TARGET_UPDATES={a.steps} GLOBAL_BATCH=2')
        print(f'TRAINING_SAMPLES_CONSUMED={a.steps*2} (dataset=289703)')
        print(f'EXPECTED_CHECKPOINTS_UPPER_ESTIMATE={saves}')
        print(f'CHECKPOINT_STORAGE_BUDGET_GIB={needed:.1f}')
        print(f'SSD_FREE_GIB={free_gib():.1f}')
        print('RESOURCE_PLAN=2x RTX A5000, four Torch workers (by default), one workstation process')
        print('PRODUCTION_APPROVED=NO')
        print('TRAINING_STARTED=NO')
    elif a.action == 'status':
        status(a.experiment, a.steps)
    elif a.action == 'stop':
        stop(a.experiment)
    elif a.action in ('start', 'resume'):
        if not a.execute:
            raise RuntimeError('Training launch refused without explicit --execute')
        submit(a)
    else:
        raise ValueError(a.action)
    return 0


if __name__ == '__main__':
    rc = 0
    try:
        rc = main()
    except Exception as exc:
        print(f'BLOCKED: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        rc = 2
    finally:
        print('#' * 110, flush=True)
        print(f'OPERATOR_EXIT_CODE={rc}', flush=True)
        print(f'OPERATOR_EVIDENCE_ROOT={OPSROOT}', flush=True)
        print('#' * 110, flush=True)
    raise SystemExit(rc)
