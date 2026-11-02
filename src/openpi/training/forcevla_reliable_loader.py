"""ForceVLA-native deterministic Torch -> JAX loader.

Preserves the original openpi dataset/transforms.
Uses deterministic epoch-aware batch sampling.

The checkpoint cursor is committed by the training loop,
never by worker prefetch.

No model, loss, optimizer or normalization modification.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import random
from pathlib import Path

import jax
import numpy as np
import torch

from openpi.training import data_loader as original

from openpi.training.forcevla_replay import (
    ReplayBatchSampler,
    ReplayContract,
)


def file_sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(4 * 1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


class EpochTaggedBatchSampler:
    """Attach epoch to each sample for stateless RNG."""

    def __init__(self, replay):
        self.replay = replay

    def __len__(self):
        return len(self.replay)

    def __iter__(self):
        c = self.replay.contract

        for offset, batch in enumerate(self.replay):
            step = self.replay.completed_updates + offset
            epoch = step // c.batches_per_epoch

            yield [
                (epoch, int(index))
                for index in batch
            ]


class SeededSampleDataset(torch.utils.data.Dataset):
    """Stable host-side random streams per epoch/sample.

    The original dataset and transforms are unchanged.
    Random streams are restored after __getitem__ so that
    num_workers=0 does not perturb the parent process RNG.
    """

    def __init__(self, dataset, contract):
        self.dataset = dataset
        self.contract = contract

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        if (
            not isinstance(key, tuple)
            or len(key) != 2
        ):
            raise TypeError(
                "Replay dataset requires (epoch, index)"
            )

        epoch, index = key

        if (
            type(epoch) is not int
            or type(index) is not int
            or epoch < 0
            or not 0 <= index < len(self.dataset)
        ):
            raise ValueError(
                f"Invalid replay sample: {key}"
            )

        material = (
            f"forcevla-sample-v1:"
            f"{self.contract.digest}:{epoch}:{index}"
        ).encode()

        seed = int.from_bytes(
            hashlib.sha256(material).digest()[:8],
            "big",
        ) & ((1 << 63) - 1)

        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()

        try:
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)

            return self.dataset[index]

        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)


class ReplayArrayLoader:
    """Adapter matching the original ForceVLA DataLoader."""

    def __init__(self, torch_loader, sharding):
        self.torch_loader = torch_loader
        self.sharding = sharding

    def __iter__(self):
        for batch in self.torch_loader:
            yield jax.tree.map(
                lambda x: jax.make_array_from_process_local_data(
                    self.sharding, x
                ),
                batch,
            )


def create_replay_data_loader(
    config,
    contract: ReplayContract,
    *,
    completed_updates: int,
    target_updates: int,
    info_path,
    norm_stats_path,
    sharding=None,
):
    """Return a standard ForceVLA DataLoaderImpl.

    The caller must bind completed_updates to the
    *restored, verified* TrainState.step.
    """

    if jax.process_count() != 1:
        raise RuntimeError(
            "Replay v1 supports one JAX process only"
        )

    if config.batch_size != contract.batch_size:
        raise RuntimeError(
            "Batch size differs from replay contract"
        )

    if config.seed != contract.seed:
        raise RuntimeError(
            "Seed differs from replay contract"
        )

    if file_sha256(info_path) != (
        contract.dataset_info_sha256
    ):
        raise RuntimeError(
            "Training metadata fingerprint mismatch"
        )

    if file_sha256(norm_stats_path) != (
        contract.norm_stats_sha256
    ):
        raise RuntimeError(
            "Normalization fingerprint mismatch"
        )

    data_config = config.data.create(
        config.assets_dirs,
        config.model,
    )

    if data_config.rlds_data_dir is not None:
        raise RuntimeError(
            "Replay loader supports Torch/LeRobot only"
        )

    dataset = original.create_torch_dataset(
        data_config,
        config.model.action_horizon,
        config.model,
    )

    dataset = original.transform_dataset(
        dataset,
        data_config,
    )

    if len(dataset) != contract.dataset_size:
        raise RuntimeError(
            "Dataset length differs from replay contract: "
            f"{len(dataset)} != {contract.dataset_size}"
        )

    seeded_dataset = SeededSampleDataset(
        dataset,
        contract,
    )

    replay = ReplayBatchSampler(
        contract,
        completed_updates=completed_updates,
        target_updates=target_updates,
    )

    tagged = EpochTaggedBatchSampler(replay)

    if sharding is None:
        sharding = jax.sharding.NamedSharding(
            jax.sharding.Mesh(
                jax.devices(), ("B",)
            ),
            jax.sharding.PartitionSpec("B"),
        )

    kwargs = {
        "dataset": seeded_dataset,
        "batch_sampler": tagged,
        "num_workers": config.num_workers,
        "collate_fn": original._collate_fn,
        "worker_init_fn": original._worker_init_fn,
    }

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    kwargs["generator"] = generator

    if config.num_workers > 0:
        kwargs["multiprocessing_context"] = (
            multiprocessing.get_context("spawn")
        )
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    torch_loader = torch.utils.data.DataLoader(
        **kwargs
    )

    adapter = ReplayArrayLoader(
        torch_loader,
        sharding,
    )

    return original.DataLoaderImpl(
        data_config,
        adapter,
    )
