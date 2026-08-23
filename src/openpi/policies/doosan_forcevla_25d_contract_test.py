import unittest

import numpy as np

from openpi.policies import forcevla_policy
from openpi.training import config as training_config


class DoosanForcevla25DContractTest(unittest.TestCase):
    def test_25d_profile_is_registered_without_changing_v1(self):
        profile = training_config.get_config("doosan_forcevla_25d_32d_compat_contract")
        old_native = training_config.get_config("doosan_forcevla_native_contract")
        old_compat = training_config.get_config("doosan_forcevla_32d_compat_contract")

        assert profile.model.state_dim == 25
        assert profile.model.proprio_dim == 19
        assert profile.model.wrench_dim == 6
        assert profile.model.state_proj_dim == 32
        assert profile.model.action_dim == 32
        assert profile.model.action_horizon == 50

        assert profile.num_train_steps == 0
        assert profile.batch_size == 1
        assert isinstance(profile.data, training_config.LeRobotDoosanForcevlaDataConfig)

        assert old_native.model.state_dim == 13
        assert old_native.model.proprio_dim == 7
        assert old_native.model.wrench_dim == 6
        assert old_native.model.action_dim == 7

        assert old_compat.model.state_dim == 13
        assert old_compat.model.proprio_dim == 7
        assert old_compat.model.wrench_dim == 6
        assert old_compat.model.state_proj_dim == 32
        assert old_compat.model.action_dim == 32

    def test_25d_adapter_preserves_state_and_pads_only_actions(self):
        state = np.arange(25, dtype=np.float32)
        actions = np.arange(
            50 * 7,
            dtype=np.float32,
        ).reshape(50, 7)

        tcp_camera = np.zeros(
            (8, 10, 3),
            dtype=np.uint8,
        )
        external_camera_2 = np.ones(
            (8, 10, 3),
            dtype=np.uint8,
        )

        transform = forcevla_policy.DoosanForcevlaInputs(
            state_dim=25,
            action_dim=32,
            robot_action_dim=7,
        )

        result = transform(
            {
                "state": state,
                "actions": actions,
                "images": {
                    "tcp_camera": tcp_camera,
                    "external_camera_2": external_camera_2,
                },
                "prompt": "insert the peg",
            }
        )

        np.testing.assert_array_equal(
            result["state"],
            state,
        )
        assert result["state"].shape == (25,)
        assert result["actions"].shape == (50, 32)

        np.testing.assert_array_equal(
            result["actions"][:, :7],
            actions,
        )
        assert np.count_nonzero(result["actions"][:, 7:]) == 0

        assert bool(result["image_mask"]["base_0_rgb"])
        assert bool(result["image_mask"]["left_wrist_0_rgb"])
        assert not bool(result["image_mask"]["right_wrist_0_rgb"])

        decoded = forcevla_policy.DoosanForcevlaOutputs(robot_action_dim=7)(
            {
                "actions": result["actions"],
            }
        )

        assert decoded["actions"].shape == (50, 7)
        np.testing.assert_array_equal(
            decoded["actions"],
            actions,
        )


if __name__ == "__main__":
    unittest.main()
