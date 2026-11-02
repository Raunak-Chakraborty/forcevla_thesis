"""Sampler replay qualification: zero/four workers and epoch wrap."""
from pathlib import Path
import sys

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1]
        / "src/openpi/training"
    ),
)

from torch.utils.data import DataLoader, Dataset

from forcevla_replay import (
    ReplayBatchSampler,
    ReplayContract,
    validate_cursor,
)


class IndexDataset(Dataset):
    def __len__(self):
        return 11

    def __getitem__(self, index):
        return int(index)


def observed(contract, begin, end, workers):
    sampler = ReplayBatchSampler(
        contract,
        completed_updates=begin,
        target_updates=end,
    )

    kwargs = {
        "dataset": IndexDataset(),
        "batch_sampler": sampler,
        "num_workers": workers,
    }

    if workers:
        kwargs.update({
            "multiprocessing_context": "spawn",
            "persistent_workers": True,
            "prefetch_factor": 2,
        })

    loader = DataLoader(**kwargs)

    return [
        tuple(map(int, batch.tolist()))
        for batch in loader
    ]


def main():
    contract = ReplayContract(
        11, 2, 42, "a" * 64, "b" * 64
    )

    uninterrupted = observed(contract, 0, 13, 0)

    assert len(uninterrupted) == 13

    assert uninterrupted == observed(
        contract, 0, 13, 4
    )

    assert uninterrupted == (
        observed(contract, 0, 4, 4)
        + observed(contract, 4, 13, 4)
    )

    assert uninterrupted == (
        observed(contract, 0, 5, 0)
        + observed(contract, 5, 13, 0)
    )

    assert uninterrupted == (
        observed(contract, 0, 10, 0)
        + observed(contract, 10, 13, 4)
    )

    assert all(
        len(batch) == 2
        for batch in uninterrupted
    )

    for start in (0, 5):
        values = [
            index
            for batch in uninterrupted[start:start + 5]
            for index in batch
        ]
        assert len(set(values)) == 10

    assert contract.cursor(5)["epoch"] == 1
    assert contract.cursor(5)["batch_in_epoch"] == 0

    validate_cursor(
        contract,
        contract.cursor(8),
        restored_step=8,
    )

    for corrupted in (
        {
            **contract.cursor(8),
            "completed_updates": 7,
        },
        {
            **contract.cursor(8),
            "contract_sha256": "c" * 64,
        },
    ):
        try:
            validate_cursor(
                contract,
                corrupted,
                restored_step=8,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                "Corrupted cursor was accepted"
            )

    try:
        ReplayBatchSampler(
            contract,
            completed_updates=14,
            target_updates=13,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Invalid resume target accepted"
        )

    print("SAMPLER_MULTI_EPOCH=PASS")
    print("SAMPLER_WORKERS_0_AND_4=PASS")
    print("SAMPLE_STREAM_RESUME=PASS")
    print("CURSOR_CORRUPTION_REJECTED=PASS")
    print("CPU_REPLAY_TEST=PASS")


if __name__ == "__main__":
    main()
