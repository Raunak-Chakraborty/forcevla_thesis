import unittest

import jax
import jax.numpy as jnp

from openpi.models import pi0_force


class Pi0ForceDimensionContractTest(unittest.TestCase):
    def test_default_contract_preserves_official_32d_shapes(self):
        config = pi0_force.Pi0_GuidanceConfig()
        observation_spec, action_spec = config.inputs_spec(batch_size=2)

        self.assertEqual(config.state_dim, 32)
        self.assertEqual(config.proprio_dim, 7)
        self.assertEqual(config.wrench_dim, 6)
        self.assertEqual(config.state_proj_dim, 32)
        self.assertEqual(config.action_dim, 32)
        self.assertEqual(config.action_horizon, 50)

        self.assertEqual(tuple(observation_spec.state.shape), (2, 32))
        self.assertEqual(tuple(action_spec.shape), (2, 50, 32))

    def test_native_doosan_contract_declares_13d_state_and_7d_action(self):
        config = pi0_force.Pi0_GuidanceConfig(
            state_dim=13,
            proprio_dim=7,
            wrench_dim=6,
            state_proj_dim=7,
            action_dim=7,
            action_horizon=50,
        )
        observation_spec, action_spec = config.inputs_spec(batch_size=3)

        self.assertEqual(tuple(observation_spec.state.shape), (3, 13))
        self.assertEqual(tuple(action_spec.shape), (3, 50, 7))

    def test_doosan_32d_compatibility_contract_keeps_semantic_state_13d(self):
        config = pi0_force.Pi0_GuidanceConfig(
            state_dim=13,
            proprio_dim=7,
            wrench_dim=6,
            state_proj_dim=32,
            action_dim=32,
            action_horizon=50,
        )
        observation_spec, action_spec = config.inputs_spec(batch_size=1)

        self.assertEqual(tuple(observation_spec.state.shape), (1, 13))
        self.assertEqual(tuple(action_spec.shape), (1, 50, 32))

    def test_invalid_state_dimension_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "state_dim"):
            pi0_force.Pi0_GuidanceConfig(
                state_dim=12,
                proprio_dim=7,
                wrench_dim=6,
            )

    def test_projection_dimension_smaller_than_proprioception_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "state_proj_dim"):
            pi0_force.Pi0_GuidanceConfig(
                state_dim=13,
                proprio_dim=7,
                wrench_dim=6,
                state_proj_dim=6,
            )

    def test_native_dummy_model_uses_independent_projection_dimensions(self):
        config = pi0_force.Pi0_GuidanceConfig(
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            state_dim=13,
            proprio_dim=7,
            wrench_dim=6,
            state_proj_dim=7,
            action_dim=7,
            action_horizon=50,
        )

        model = config.create(jax.random.key(0))

        self.assertEqual(model.state_proj.in_features, 7)
        self.assertEqual(model.force_in_proj.in_features, 6)
        self.assertEqual(model.action_in_proj.in_features, 7)
        self.assertEqual(model.action_out_proj.out_features, 7)

        observation = config.fake_obs(batch_size=2)
        actions = config.fake_act(batch_size=2)
        timestep = jnp.full((2,), 0.5, dtype=jnp.float32)

        suffix_tokens, suffix_mask, suffix_ar_mask, force_tokens = (
            model.embed_suffix(observation, actions, timestep)
        )

        self.assertEqual(
            tuple(suffix_tokens.shape[:2]),
            (2, 1 + config.action_horizon),
        )
        self.assertEqual(
            tuple(suffix_mask.shape),
            (2, 1 + config.action_horizon),
        )
        self.assertEqual(
            tuple(suffix_ar_mask.shape),
            (1 + config.action_horizon,),
        )
        self.assertEqual(tuple(force_tokens.shape[:2]), (2, 1))
        self.assertEqual(
            suffix_tokens.shape[-1],
            model.state_proj.out_features,
        )
        self.assertEqual(
            force_tokens.shape[-1],
            model.force_in_proj.out_features,
        )


if __name__ == "__main__":
    unittest.main()
