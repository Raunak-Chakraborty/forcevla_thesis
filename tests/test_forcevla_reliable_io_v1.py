"""CPU-only integrity and journal negative tests; no training or heavyweight checkpoint."""
import importlib.util
import json
import tempfile
from pathlib import Path

SRC = Path('/home/ktt_rc/robotics_thesis/forcevla_lab/worktrees/forcevla_lora_reliability_v1/scripts/train_forcevla_reliable_v1.py')
spec = importlib.util.spec_from_file_location('fvreliable', SRC)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class FakeManager:
    def wait_until_finished(self):
        pass
    def all_steps(self):
        return [2]

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    cp = root / 'checkpoints' / '2'
    (cp / 'train_state').mkdir(parents=True)
    (cp / 'params').mkdir()
    (cp / '_CHECKPOINT_METADATA').write_text('{"finalized":true}\n')
    (cp / 'train_state' / 'payload').write_bytes(b'optimizer-and-step-example')
    (cp / 'params' / 'payload').write_bytes(b'model-example')
    run = root / 'run'
    run.mkdir()
    receipt = m.publish_committed(FakeManager(), cp.parent, run, 2)
    m.validate_receipt(cp, receipt)
    assert json.loads((run / 'latest_committed.json').read_text()) == receipt
    try:
        m.atomic_json(run / 'receipts' / 'step-00000002.json', receipt, exclusive=True)
    except FileExistsError:
        pass
    else:
        raise AssertionError('Receipt overwrite was accepted')
    telemetry = run / 'telemetry.jsonl'
    m.log_event(telemetry, event='update', completed_updates=2)
    assert json.loads(telemetry.read_text())['completed_updates'] == 2
    (cp / 'params' / 'payload').write_bytes(b'CORRUPTED')
    try:
        m.validate_receipt(cp, receipt)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Corrupt checkpoint was accepted')

print('ATOMIC_RECEIPT=PASS')
print('CHECKPOINT_TAMPER_REJECTED=PASS')
print('STEP_JOURNAL=PASS')
print('CPU_IO_TEST=PASS')
