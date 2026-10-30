import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.shared import nnx_utils

from forcevla_validation_bridge_v1 import (
    load_validation,
    check_sample,
)
from forcevla_offline_eval_v1 import (
    CKPT,
    make_observation,
)

POSITIONS = (0.2, 0.5, 0.8)
SEEDS = 2
EPISODES = tuple(range(255, 289))


def main(output):
    assert jax.device_count() == 1
    assert jax.devices()[0].platform == "gpu"
    assert (CKPT / "_CHECKPOINT_METADATA").is_file()

    transformed, _, config = load_validation()

    # Recover original unfiltered LeRobot dataset.
    raw = transformed._dataset._dataset

    assert len(raw.episode_data_index["from"]) == 323

    by_episode = defaultdict(list)

    for index, episode in enumerate(
        raw.hf_dataset["episode_index"]
    ):
        episode = int(episode)
        if episode in EPISODES:
            by_episode[episode].append(index)

    assert set(by_episode) == set(EPISODES)
    assert sum(map(len, by_episode.values())) == 34652

    print("VALIDATION_ISOLATION=PASS", flush=True)

    selected = []

    for episode in EPISODES:
        indices = by_episode[episode]

        for position_number, fraction in enumerate(
            POSITIONS
        ):
            index = indices[
                int(fraction * (len(indices) - 1))
            ]

            sample = transformed[index]
            state, actions = check_sample(sample)

            selected.append({
                "episode": episode,
                "position_number": position_number,
                "frame_index": index,
                "sample": sample,
                "wrench": np.asarray(
                    state[19:25], dtype=np.float32
                ).copy(),
                "truth": np.asarray(
                    actions[:, :7], dtype=np.float32
                ).copy(),
            })

    assert len(selected) == 102

    print("DISTINCT_OBSERVATIONS=102", flush=True)
    print("SAMPLING_SEEDS_PER_OBSERVATION=2", flush=True)

    params = model_lib.restore_params(
        CKPT / "params"
    )
    model = config.model.load(params)
    model.eval()

    inference = nnx_utils.module_jit(
        model.sample_actions
    )

    print("MODEL_RESTORE=PASS", flush=True)

    rows = []

    for episode_number, episode in enumerate(EPISODES):
        for position_number in range(len(POSITIONS)):
            slot = (
                episode_number * len(POSITIONS)
                + position_number
            )

            item = selected[slot]
            observation = make_observation(
                item["sample"]
            )

            # Shuffle wrench across DIFFERENT episodes,
            # retaining the same relative trajectory position.
            other_episode = (
                episode_number + 17
            ) % len(EPISODES)

            other_slot = (
                other_episode * len(POSITIONS)
                + position_number
            )

            shuffled_wrench = selected[
                other_slot
            ]["wrench"]

            no_force = observation.replace(
                state=observation.state.at[
                    :, 19:25
                ].set(0.0)
            )

            shuffled = observation.replace(
                state=observation.state.at[
                    :, 19:25
                ].set(jnp.asarray(shuffled_wrench))
            )

            for repetition in range(SEEDS):
                seed = jax.random.key(
                    100000
                    + episode * 100
                    + position_number * 10
                    + repetition
                )

                def predict(obs):
                    result = inference(seed, obs)
                    result = np.asarray(
                        jax.device_get(result)
                    )
                    assert result.shape == (
                        1, 50, 32
                    )
                    assert np.isfinite(result).all()
                    return result[0, :, :7]

                actual = predict(observation)
                zero = predict(no_force)
                shuffled_pred = predict(shuffled)

                truth = item["truth"]

                row = {
                    "episode": episode,
                    "position": position_number,
                    "seed": repetition,
                    "mae_actual": float(
                        np.mean(np.abs(
                            actual - truth
                        ))
                    ),
                    "mae_zero": float(
                        np.mean(np.abs(
                            zero - truth
                        ))
                    ),
                    "mae_shuffled": float(
                        np.mean(np.abs(
                            shuffled_pred - truth
                        ))
                    ),
                    "first_mae_actual": float(
                        np.mean(np.abs(
                            actual[0] - truth[0]
                        ))
                    ),
                    "first_mae_zero": float(
                        np.mean(np.abs(
                            zero[0] - truth[0]
                        ))
                    ),
                    "sensitivity_zero": float(
                        np.mean(np.abs(
                            actual - zero
                        ))
                    ),
                    "sensitivity_shuffled": float(
                        np.mean(np.abs(
                            actual - shuffled_pred
                        ))
                    ),
                }

                assert all(
                    math.isfinite(value)
                    for key, value in row.items()
                    if key not in (
                        "episode", "position", "seed"
                    )
                )

                rows.append(row)

        print(
            f"EPISODES_COMPLETED={episode_number + 1}/34",
            flush=True,
        )

    assert len(rows) == 204

    metrics = [
        "mae_actual",
        "mae_zero",
        "mae_shuffled",
        "first_mae_actual",
        "first_mae_zero",
        "sensitivity_zero",
        "sensitivity_shuffled",
    ]

    summary = {
        key: round(float(np.mean([
            row[key] for row in rows
        ])), 6)
        for key in metrics
    }

    summary.update({
        "checkpoint_step": 1999,
        "episodes": 34,
        "distinct_observations": 102,
        "paired_seed_trials": 204,
        "training_performed": False,
    })

    summary["actual_minus_zero_mae"] = round(
        summary["mae_actual"]
        - summary["mae_zero"],
        6,
    )

    per_episode = {}

    for episode in EPISODES:
        subset = [
            row for row in rows
            if row["episode"] == episode
        ]

        per_episode[str(episode)] = {
            key: round(float(np.mean([
                row[key] for row in subset
            ])), 6)
            for key in metrics
        }

    output = Path(output)
    with output.open("x") as handle:
        json.dump({
            "summary": summary,
            "per_episode": per_episode,
            "trials": rows,
        }, handle, indent=2)

    print(
        "STRATIFIED_SUMMARY="
        + json.dumps(summary, sort_keys=True),
        flush=True,
    )
    print(f"RESULT_FILE={output}", flush=True)
    print("STAGE_21=PASS", flush=True)


if __name__ == "__main__":
    assert len(sys.argv) == 2
    main(sys.argv[1])
