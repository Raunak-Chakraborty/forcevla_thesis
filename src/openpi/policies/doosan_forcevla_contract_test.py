import ast
import inspect
import textwrap
import unittest

import numpy as np

from openpi.policies import forcevla_policy
from openpi.training import config as training_config


class DoosanForcevlaContractTest(unittest.TestCase):
    def setUp(self):
        self.state = np.arange(13, dtype=np.float32)
        self.actions = np.arange(
            50 * 7,
            dtype=np.float32,
        ).reshape(50, 7)

        self.external_camera_1 = np.full(
            (8, 10, 3),
            11,
            dtype=np.uint8,
        )
        self.tcp_camera = np.full(
            (3, 8, 10),
            0.5,
            dtype=np.float32,
        )
        self.external_camera_2 = np.full(
            (8, 10, 3),
            33,
            dtype=np.uint8,
        )

    def make_data(self):
        return {
            "state": self.state,
            "actions": self.actions,
            "images": {
                "external_camera_1": self.external_camera_1,
                "tcp_camera": self.tcp_camera,
                "external_camera_2": self.external_camera_2,
            },
            "prompt": "insert the peg",
        }

    def test_native_adapter_preserves_state_and_actions(self):
        transform = forcevla_policy.DoosanForcevlaInputs(
            state_dim=13,
            action_dim=7,
        )

        result = transform(self.make_data())

        np.testing.assert_array_equal(
            result["state"],
            self.state,
        )
        np.testing.assert_array_equal(
            result["actions"],
            self.actions,
        )

        self.assertEqual(result["state"].shape, (13,))
        self.assertEqual(result["actions"].shape, (50, 7))

    def test_two_real_cameras_map_to_model_slots(self):
        transform = forcevla_policy.DoosanForcevlaInputs(
            state_dim=13,
            action_dim=7,
        )

        data = self.make_data()
        del data["images"]["external_camera_1"]

        result = transform(data)

        np.testing.assert_array_equal(
            result["image"]["base_0_rgb"],
            forcevla_policy._parse_image(self.external_camera_2),
        )

        np.testing.assert_array_equal(
            result["image"]["left_wrist_0_rgb"],
            forcevla_policy._parse_image(self.tcp_camera),
        )

        self.assertEqual(
            result["image"]["left_wrist_0_rgb"].shape,
            (8, 10, 3),
        )
        self.assertEqual(
            result["image"]["left_wrist_0_rgb"].dtype,
            np.uint8,
        )

        np.testing.assert_array_equal(
            result["image"]["right_wrist_0_rgb"],
            np.zeros_like(
                forcevla_policy._parse_image(
                    self.external_camera_2
                )
            ),
        )

        self.assertTrue(
            bool(result["image_mask"]["base_0_rgb"])
        )
        self.assertTrue(
            bool(result["image_mask"]["left_wrist_0_rgb"])
        )
        self.assertFalse(
            bool(result["image_mask"]["right_wrist_0_rgb"])
        )

    def test_32d_compatibility_pads_only_actions(self):
        transform = forcevla_policy.DoosanForcevlaInputs(
            state_dim=13,
            action_dim=32,
        )

        result = transform(self.make_data())

        self.assertEqual(result["state"].shape, (13,))
        self.assertEqual(result["actions"].shape, (50, 32))

        np.testing.assert_array_equal(
            result["actions"][:, :7],
            self.actions,
        )
        self.assertEqual(
            np.count_nonzero(result["actions"][:, 7:]),
            0,
        )

    def test_missing_camera_is_rejected(self):
        transform = forcevla_policy.DoosanForcevlaInputs(
            state_dim=13,
            action_dim=7,
        )

        for camera_name in (
            "tcp_camera",
            "external_camera_2",
        ):
            with self.subTest(camera_name=camera_name):
                data = self.make_data()
                del data["images"]["external_camera_1"]
                del data["images"][camera_name]

                with self.assertRaisesRegex(
                    KeyError,
                    camera_name,
                ):
                    transform(data)

    def test_wrong_dimensions_are_rejected(self):
        transform = forcevla_policy.DoosanForcevlaInputs(
            state_dim=13,
            action_dim=7,
        )

        data = self.make_data()
        data["state"] = np.zeros((12,), dtype=np.float32)

        with self.assertRaisesRegex(
            ValueError,
            "state dimension 13",
        ):
            transform(data)

        data = self.make_data()
        data["actions"] = np.zeros(
            (50, 6),
            dtype=np.float32,
        )

        with self.assertRaisesRegex(
            ValueError,
            "action dimension 7",
        ):
            transform(data)

    def test_output_adapter_returns_seven_channels(self):
        predicted = np.arange(
            50 * 32,
            dtype=np.float32,
        ).reshape(50, 32)

        result = forcevla_policy.DoosanForcevlaOutputs()(
            {"actions": predicted}
        )

        np.testing.assert_array_equal(
            result["actions"],
            predicted[:, :7],
        )

    def test_registered_contract_configurations(self):
        native = training_config.get_config(
            "doosan_forcevla_native_contract"
        )
        compat = training_config.get_config(
            "doosan_forcevla_32d_compat_contract"
        )

        self.assertEqual(native.model.state_dim, 13)
        self.assertEqual(native.model.state_proj_dim, 7)
        self.assertEqual(native.model.action_dim, 7)
        self.assertEqual(native.model.action_horizon, 50)

        self.assertEqual(compat.model.state_dim, 13)
        self.assertEqual(compat.model.state_proj_dim, 32)
        self.assertEqual(compat.model.action_dim, 32)
        self.assertEqual(compat.model.action_horizon, 50)

        self.assertEqual(native.num_train_steps, 0)
        self.assertEqual(compat.num_train_steps, 0)

        self.assertIsInstance(
            native.data,
            training_config.LeRobotDoosanForcevlaDataConfig,
        )

    def test_doosan_factory_has_no_delta_action_transform(self):
        source = textwrap.dedent(
            inspect.getsource(
                training_config
                .LeRobotDoosanForcevlaDataConfig
                .create
            )
        )

        tree = ast.parse(source)
        called_names = set()

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            function = node.func

            if isinstance(function, ast.Name):
                called_names.add(function.id)
            elif isinstance(function, ast.Attribute):
                called_names.add(function.attr)

        self.assertNotIn("DeltaActions", called_names)
        self.assertNotIn("AbsoluteActions", called_names)

        string_constants = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
        }

        self.assertNotIn(
            "observation.images.external_camera_1",
            string_constants,
        )
        self.assertIn(
            "observation.images.tcp_camera",
            string_constants,
        )
        self.assertIn(
            "observation.images.external_camera_2",
            string_constants,
        )


if __name__ == "__main__":
    unittest.main()
