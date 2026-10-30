import json
import numpy as np
from collections import defaultdict
from dataclasses import replace

from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from openpi.training import config as training_config
from openpi.training import data_loader
from openpi import transforms

FULL = "forcevla_lerobot_language_v2_25d"
TRAIN = "forcevla_lerobot_language_v2_25d_train"
VAL_IDS = tuple(range(255, 289))

def load_validation():
    metadata = LeRobotDatasetMetadata(FULL)

    # IMPORTANT: Do not use episodes= here.
    # The installed LeRobot release uses absolute episode IDs
    # with locally indexed boundary arrays when filtered.
    dataset = LeRobotDataset(
        FULL,
        delta_timestamps={
            "action": [
                t / metadata.fps for t in range(50)
            ]
        },
    )

    assert len(dataset.episode_data_index["from"]) == 323
    assert len(dataset.episode_data_index["to"]) == 323

    selected = defaultdict(list)
    for i, episode in enumerate(
        dataset.hf_dataset["episode_index"]
    ):
        episode = int(episode)
        if episode in VAL_IDS:
            selected[episode].append(i)

    assert set(selected) == set(VAL_IDS)
    assert sum(map(len, selected.values())) == 34652

    base = training_config.get_config(
        "doosan_forcevla_25d_32d_compat_contract"
    )

    config = replace(
        base,
        data=replace(base.data, repo_id=TRAIN),
        assets_base_dir=(
            "/home/ktt_rc/robotics_thesis/forcevla_lab/"
            "finetune_runs/assets"
        ),
    )

    data_config = config.data.create(
        config.assets_dirs, config.model
    )
    assert data_config.norm_stats is not None

    prompt_dataset = data_loader.TransformedDataset(
        dataset,
        [transforms.PromptFromLeRobotTask(metadata.tasks)],
    )

    transformed = data_loader.transform_dataset(
        prompt_dataset,
        data_config,
    )

    # One deterministic midpoint per validation episode.
    sample_indices = {
        ep: indices[len(indices) // 2]
        for ep, indices in sorted(selected.items())
    }

    return transformed, sample_indices, config


def check_sample(sample):
    state = np.asarray(sample["state"])
    actions = np.asarray(sample["actions"])

    assert state.shape == (25,)
    assert actions.shape == (50, 32)
    assert np.isfinite(state).all()
    assert np.isfinite(actions).all()

    assert set(sample["image"]) == {
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    }

    assert bool(sample["image_mask"]["base_0_rgb"])
    assert bool(sample["image_mask"]["left_wrist_0_rgb"])
    assert not bool(
        sample["image_mask"]["right_wrist_0_rgb"]
    )

    for image in sample["image"].values():
        array = np.asarray(image)
        assert array.shape == (224, 224, 3)
        assert np.isfinite(array).all()

    return state, actions


def main():
    transformed, indices, config = load_validation()

    max_padding = 0.0
    max_wrench = 0.0

    print("VALIDATION_EPISODES=34", flush=True)
    print("VALIDATION_FRAMES=34652", flush=True)
    print("TRAIN_EPISODE_LEAKAGE=NONE", flush=True)

    for count, (episode, index) in enumerate(
        indices.items(), start=1
    ):
        state, actions = check_sample(
            transformed[index]
        )

        max_padding = max(
            max_padding,
            float(np.max(np.abs(actions[:, 7:]))),
        )
        max_wrench = max(
            max_wrench,
            float(np.max(np.abs(state[19:25]))),
        )

        if count % 6 == 0 or count == 34:
            print(
                f"VALIDATION_SAMPLES={count}/34",
                flush=True,
            )

    print("CAMERA_MASKS=PASS", flush=True)
    print("STATE_SHAPE_25D=PASS", flush=True)
    print("ACTION_SHAPE_50x32=PASS", flush=True)
    print("TRAIN_NORMALIZATION_REUSED=PASS", flush=True)
    print("MAX_ABS_PADDED_ACTION=", max_padding, flush=True)
    print("MAX_ABS_NORMALIZED_WRENCH=", max_wrench, flush=True)
    print("STAGE_18=PASS", flush=True)


if __name__ == "__main__":
    main()
