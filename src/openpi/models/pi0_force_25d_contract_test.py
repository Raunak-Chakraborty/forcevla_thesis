import unittest

import jax
import numpy as np

from openpi.models import pi0_force


class Pi0Force25DContractTest(unittest.TestCase):
    def make_config(self):
        return pi0_force.Pi0_GuidanceConfig(
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            state_dim=25,
            proprio_dim=19,
            wrench_dim=6,
            state_proj_dim=32,
            action_dim=32,
            action_horizon=50,
        )

    def test_25d_contract_dimensions(self):
        config = self.make_config()
        observation_spec, action_spec = config.inputs_spec(batch_size=2)

        assert config.state_dim == 25
        assert config.proprio_dim == 19
        assert config.wrench_dim == 6
        assert config.proprio_dim + config.wrench_dim == config.state_dim
        assert config.state_proj_dim == 32
        assert config.action_dim == 32
        assert config.action_horizon == 50

        assert tuple(observation_spec.state.shape) == (2, 25)
        assert tuple(action_spec.shape) == (2, 50, 32)

    def test_25d_dummy_runtime_is_finite(self):
        config = self.make_config()
        model = config.create(jax.random.key(0))

        assert model.state_proj.in_features == 32
        assert model.force_in_proj.in_features == 6
        assert model.action_in_proj.in_features == 32
        assert model.action_out_proj.out_features == 32

        observation = config.fake_obs(batch_size=1)
        actions = config.fake_act(batch_size=1)

        loss = model.compute_loss(
            jax.random.key(1),
            observation,
            actions,
            train=False,
        )
        loss_array = np.asarray(jax.device_get(loss))

        assert loss_array.shape == (1, 50)
        assert np.isfinite(loss_array).all()

        sampled = model.sample_actions(
            jax.random.key(2),
            observation,
            num_steps=1,
        )
        sampled_array = np.asarray(jax.device_get(sampled))

        assert sampled_array.shape == (1, 50, 32)
        assert np.isfinite(sampled_array).all()
        assert sampled_array[..., :7].shape == (1, 50, 7)


if __name__ == "__main__":
    unittest.main()
