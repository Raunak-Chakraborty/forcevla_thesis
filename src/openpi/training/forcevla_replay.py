"""Deterministic ForceVLA Torch-batch replay.

Only completed JAX optimizer updates determine the saved cursor.
DataLoader prefetch must never advance the checkpoint cursor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import torch
from torch.utils.data import Sampler


@dataclass(frozen=True)
class ReplayContract:
    dataset_size: int
    batch_size: int
    seed: int
    dataset_info_sha256: str
    norm_stats_sha256: str

    def __post_init__(self):
        for key in ("dataset_size", "batch_size", "seed"):
            value = getattr(self, key)
            lower = 0 if key == "seed" else 1
            if type(value) is not int or value < lower:
                raise ValueError(f"Invalid {key}: {value!r}")

        if self.dataset_size < self.batch_size:
            raise ValueError("Dataset smaller than global batch")

        for key in ("dataset_info_sha256", "norm_stats_sha256"):
            value = getattr(self, key)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError(f"Invalid {key}")

    @property
    def batches_per_epoch(self):
        # Consistent with drop_last=True.
        return self.dataset_size // self.batch_size

    @property
    def digest(self):
        encoded = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def cursor(self, completed_updates):
        if (
            type(completed_updates) is not int
            or completed_updates < 0
        ):
            raise ValueError("Invalid optimizer update count")

        epoch, batch = divmod(
            completed_updates, self.batches_per_epoch
        )

        return {
            "schema": "forcevla-torch-batch-replay-v1",
            "contract_sha256": self.digest,
            "completed_updates": completed_updates,
            "epoch": epoch,
            "batch_in_epoch": batch,
            "sample_offset_in_epoch": batch * self.batch_size,
        }


def validate_cursor(contract, cursor, *, restored_step):
    if (
        not isinstance(cursor, dict)
        or cursor != contract.cursor(restored_step)
    ):
        raise RuntimeError(
            "Resume blocked: sampler cursor or "
            "scientific contract mismatch"
        )


class ReplayBatchSampler(Sampler[list[int]]):
    """Replay batch IDs from a committed optimizer boundary.

    Supports epoch rollover and drop_last semantics.
    Prefetched samples do not change the committed cursor.
    """

    def __init__(
        self, contract, *, completed_updates, target_updates
    ):
        if type(target_updates) is not int or target_updates < 1:
            raise ValueError("Invalid target_updates")

        contract.cursor(completed_updates)

        if completed_updates > target_updates:
            raise ValueError(
                "Completed updates exceed target"
            )

        self.contract = contract
        self.completed_updates = completed_updates
        self.target_updates = target_updates

    def __len__(self):
        return self.target_updates - self.completed_updates

    def __iter__(self):
        c = self.contract
        last_epoch = None
        order = None

        for step in range(
            self.completed_updates, self.target_updates
        ):
            epoch, batch = divmod(
                step, c.batches_per_epoch
            )

            if epoch != last_epoch:
                material = (
                    f"forcevla-replay-v1:{c.seed}:{epoch}"
                ).encode()

                epoch_seed = int.from_bytes(
                    hashlib.sha256(material).digest()[:8],
                    "big",
                )

                generator = torch.Generator()
                generator.manual_seed(epoch_seed)

                order = torch.randperm(
                    c.dataset_size,
                    generator=generator,
                ).tolist()

                last_epoch = epoch

            start = batch * c.batch_size

            yield order[
                start : start + c.batch_size
            ]
