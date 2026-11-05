import argparse
import json
import math
import time
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.shared import nnx_utils

from forcevla_validation_bridge_v1 import (
    load_validation,
    check_sample,
)

ROOT = Path("/home/ktt_rc/robotics_thesis/forcevla_lab")
DATA = Path("/mnt/robotics_ssd/ktt_rc_robotics_thesis/dataset")

EXP = "lora_force_first_2000_20261030_174216"

CKPT = (
    DATA
    / "forcevla_mock_runs"
    / "checkpoints"
    / "doosan_forcevla_25d_32d_compat_contract"
    / EXP
    / "1999"
)

def make_observation(sample):
    images = {}

    for name, value in sample["image"].items():
        pixels = np.asarray(value)

        assert pixels.shape == (224, 224, 3)

        if pixels.dtype == np.uint8:
            # Match the training Observation.from_dict conversion.
            pixels = (
                pixels.astype(np.float32) / 255.0 * 2.0 - 1.0
            )
        elif np.issubdtype(pixels.dtype, np.floating):
            pixels = pixels.astype(np.float32, copy=False)
            assert pixels.min() >= -1.00001
            assert pixels.max() <= 1.00001
        else:
            raise TypeError(
                f"Unsupported image dtype: {pixels.dtype}"
            )

        assert np.isfinite(pixels).all()

        images[name] = jnp.asarray(
            pixels[None],
            dtype=jnp.float32,
        )
    masks = {
        name: jnp.asarray([bool(value)])
        for name, value in sample["image_mask"].items()
    }
    return model_lib.Observation(
        images=images,
        image_masks=masks,
        state=jnp.asarray(
            np.asarray(sample["state"])[None],
            dtype=jnp.float32,
        ),
        tokenized_prompt=jnp.asarray(
            np.asarray(sample["tokenized_prompt"])[None],
            dtype=jnp.int32,
        ),
        tokenized_prompt_mask=jnp.asarray(
            np.asarray(sample["tokenized_prompt_mask"])[None],
        ),
    )


def evaluate(count):
    assert count in (6, 34)
    assert jax.device_count() == 1
    assert jax.devices()[0].platform == "gpu"

    assert (CKPT / "_CHECKPOINT_METADATA").is_file()
    assert (CKPT / "params").is_dir()

    print("MODEL_CHECKPOINT=PASS", flush=True)

    transformed, indices, config = load_validation()

    selected = list(indices.items())[:count]

    assert len(selected) == count

    print(f"EVALUATION_EPISODES={count}", flush=True)
    print("EVALUATION_TRAINING=DISABLED", flush=True)

    params = model_lib.restore_params(
        CKPT / "params"
    )
    model = config.model.load(params)
    model.eval()

    inference = nnx_utils.module_jit(
        model.sample_actions
    )

    print("MODEL_RESTORE=PASS", flush=True)

    results = []

    for number, (episode, index) in enumerate(
        selected, start=1
    ):
        sample = transformed[index]
        _, ground_truth = check_sample(sample)

        observation = make_observation(sample)

        # Normalized zero corresponds to the training
        # wrench mean, not necessarily a physical 0 N.
        masked_state = observation.state.at[
            :, 19:25
        ].set(0.0)

        masked_observation = observation.replace(
            state=masked_state
        )

        seed = jax.random.key(42000 + episode)

        start = time.perf_counter()

        actual = np.asarray(
            jax.device_get(
                inference(seed, observation)
            )
        )[0]

        masked = np.asarray(
            jax.device_get(
                inference(seed, masked_observation)
            )
        )[0]

        assert actual.shape == (50, 32)
        assert masked.shape == (50, 32)
        assert np.isfinite(actual).all()
        assert np.isfinite(masked).all()

        # Only the 7 semantic Doosan channels.
        truth = ground_truth[:, :7]
        prediction = actual[:, :7]
        no_force = masked[:, :7]

        row = {
            "episode": episode,
            "mae_actual": float(
                np.mean(np.abs(prediction - truth))
            ),
            "mae_masked": float(
                np.mean(np.abs(no_force - truth))
            ),
            "force_sensitivity": float(
                np.mean(np.abs(prediction - no_force))
            ),
            "first_action_sensitivity": float(
                np.mean(np.abs(
                    prediction[0] - no_force[0]
                ))
            ),
            "seconds": round(
                time.perf_counter() - start, 3
            ),
        }

        assert all(
            math.isfinite(value)
            for key, value in row.items()
            if isinstance(value, float)
        )

        results.append(row)

        print(
            "EPISODE_RESULT="
            + json.dumps(row, sort_keys=True),
            flush=True,
        )

    def average(key):
        return round(float(np.mean(
            [row[key] for row in results]
        )), 6)

    summary = {
        "checkpoint_step": 1999,
        "episodes_evaluated": count,
        "semantic_action_dims": 7,
        "mae_actual": average("mae_actual"),
        "mae_masked": average("mae_masked"),
        "force_sensitivity": average(
            "force_sensitivity"
        ),
        "first_action_sensitivity": average(
            "first_action_sensitivity"
        ),
        "training_performed": False,
    }

    out = (
        ROOT / "logs/finetune_readiness/"
        f"force_eval_{count}ep_step1999.json"
    )

    assert not out.exists(), (
        f"Refusing to overwrite {out}"
    )

    out.write_text(
        json.dumps(
            {"summary": summary, "episodes": results},
            indent=2,
        )
    )

    print(
        "EVAL_SUMMARY="
        + json.dumps(summary, sort_keys=True),
        flush=True,
    )
    print(f"EVAL_RESULTS={out}", flush=True)
    print("EVALUATION=PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes", type=int, choices=[6, 34],
        required=True
    )
    args = parser.parse_args()
    evaluate(args.episodes)
