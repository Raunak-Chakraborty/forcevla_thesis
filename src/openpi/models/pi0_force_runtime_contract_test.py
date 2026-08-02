import unittest

import jax
import numpy as np

from openpi.models import pi0_force


class Pi0ForceRuntimeContractTest(unittest.TestCase):
    def test_dummy_loss_and_cached_sampling_are_finite(self):
        config = pi0_force.Pi0_GuidanceConfig(
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            state_dim=13,
            proprio_dim=7,
            wrench_dim=6,
            state_proj_dim=32,
            action_dim=32,
            action_horizon=50,
        )

        model = config.create(jax.random.key(0))
        observation = config.fake_obs(batch_size=1)
        actions = config.fake_act(batch_size=1)

        loss = model.compute_loss(
            jax.random.key(1),
            observation,
            actions,
            train=False,
        )

        loss_array = np.asarray(jax.device_get(loss))

        self.assertEqual(loss_array.shape, (1, 50))
        self.assertTrue(np.isfinite(loss_array).all())

        sampled_actions = model.sample_actions(
            jax.random.key(2),
            observation,
            num_steps=1,
        )

        sampled_array = np.asarray(
            jax.device_get(sampled_actions)
        )

        self.assertEqual(
            sampled_array.shape,
            (1, 50, 32),
        )
        self.assertTrue(np.isfinite(sampled_array).all())

        semantic_actions = sampled_array[..., :7]

        self.assertEqual(
            semantic_actions.shape,
            (1, 50, 7),
        )
        self.assertTrue(np.isfinite(semantic_actions).all())


if __name__ == "__main__":
    unittest.main()
