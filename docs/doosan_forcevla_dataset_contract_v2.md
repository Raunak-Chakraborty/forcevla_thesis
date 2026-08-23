# Doosan ForceVLA Dataset Contract v2

Contract ID: `doosan_forcevla_dataset_contract_v2`

Status: opt-in 25D Doosan M1013 ForceVLA contract.

Contract v1 remains frozen and unchanged. This v2 contract introduces a
new 25D observation-state profile while retaining the two-camera mapping,
7D semantic Cartesian action, and 50-step action horizon.

## 1. Required model-facing fields

Each aligned sample provides:

- `observation.images.tcp_camera`
- `observation.images.external_camera_2`
- `observation.state`
- `action`
- `prompt`

Episode identity and timing must preserve:

- `episode_index`
- `frame_index`
- `timestamp`

## 2. Production cameras

Only two physical RGB cameras are model inputs.

### tcp_camera

- Intel RealSense D405
- serial `409122273671`
- TCP / wrist view
- raw RGB: 640 x 480 at approximately 30 Hz
- image topic: `/doosan_cameras/tcp_camera/color/image_raw`

### external_camera_2

- Intel RealSense D435I
- serial `233622070850`
- fixed external workspace view
- raw RGB: 848 x 480 at approximately 60 Hz
- image topic: `/doosan_cameras/external_camera_2/color/image_raw`

`external_camera_1` is excluded and must not be fabricated.

The final model-facing timeline is 30 Hz with `tcp_camera` as the
reference visual timeline. Offline synchronization is owned by
`doosan-forcevla-data-tools`.

## 3. ForceVLA image-slot mapping

- `external_camera_2` -> `base_0_rgb`, mask true
- `tcp_camera` -> `left_wrist_0_rgb`, mask true
- no third physical camera -> `right_wrist_0_rgb`, zero image, mask false

The zero third image is created inside ForceVLA, not by the dataset.

## 4. Semantic observation state

`observation.state` has exactly 25 finite float channels.

Ordering:

- 0..2: TCP position `[x, y, z]`
- 3..5: TCP absolute rotation vector `[rx, ry, rz]`
- 6: measured normalized gripper open fraction
- 7..12: joint positions J1..J6
- 13..18: joint velocities J1..J6
- 19..24: external TCP wrench `[Fx, Fy, Fz, Tx, Ty, Tz]`

Equivalent grouping:

`[tcp_position_3, tcp_rotvec_3, gripper_1, joint_position_6,
joint_velocity_6, wrench_6]`

The first 19 channels are ForceVLA proprioception. The final six
channels are the dedicated wrench input.

Physical conventions:

- TCP position: absolute, robot base frame, metres
- TCP orientation: absolute base-to-TCP SO(3) rotation vector, radians
- gripper: normalized measured opening, 0 = closed, 1 = open
- joint positions: radians
- joint velocities: radians per second
- wrench order: `[Fx, Fy, Fz, Tx, Ty, Tz]`
- wrench: controller-native estimated external TCP wrench expressed
  with respect to robot base coordinates
- force: newtons
- torque: newton-metres

Wrench compensation/tare provenance must be explicit. A dataset marked
as controller-reset compensated must not receive a second implicit tare.

## 5. Semantic action

`action` remains exactly 7 finite float channels:

- 0..2: measured delta TCP translation
- 3..5: measured relative TCP rotation vector
- 6: absolute normalized gripper target open fraction

Translation:

`delta_p = p(t+1) - p(t)`

in robot-base coordinates and metres.

Rotation:

`delta_R = R(t+1) @ R(t).T`

with `Log(delta_R)` stored as a 3D rotation vector in radians.

Gripper action:

- 0 = fully closed
- 1 = fully open

The first six action channels are already Cartesian delta actions.
ForceVLA must not apply a second `DeltaActions` operation.

## 6. External versus internal dimensions

Dataset-facing dimensions:

- observation state: 25
- semantic action: 7

Profile:

`doosan_forcevla_25d_32d_compat_contract`

ForceVLA model configuration:

- `state_dim = 25`
- `proprio_dim = 19`
- `wrench_dim = 6`
- `state_proj_dim = 32`
- internal `action_dim = 32`
- `action_horizon = 50`

The converter must not pad the 7D semantic action to 32D.
Action padding and output slicing are ForceVLA responsibilities.

## 7. Normalization

The converter exports physical semantic units.

Normalization is performed by the ForceVLA/OpenPI training pipeline and
must not be permanently baked into the dataset.

Final normalization statistics must be computed using training episodes
only.

## 8. Rejection conditions

A Contract-v2 sample is invalid if:

- either required physical camera is absent
- state dimension is not exactly 25
- action dimension is not exactly 7
- any state or action value is NaN or infinity
- state ordering is ambiguous
- action semantics are ambiguous
- units or frames are ambiguous
- timestamp alignment is ambiguous
- episode boundaries are ambiguous
- Cartesian actions are differenced a second time
- a controller-reset-compensated wrench is silently re-tared

## 9. Backward compatibility

Contract v1 and its 13D ForceVLA configurations remain unchanged.

The 25D profile is opt-in and must not alter the behavior or dimensions
of:

- `doosan_forcevla_native_contract`
- `doosan_forcevla_32d_compat_contract`

The core `Pi0_Guidance` implementation and the generic
`DoosanForcevlaInputs` / `DoosanForcevlaOutputs` adapters are shared
between v1 and v2.

## 10. Ownership boundary

`doosan-forcevla-data-tools` owns MCAP decoding, synchronization,
physical-unit conversion, 25D state construction, 7D semantic action
construction, episode/frame indexing, LeRobot export, and conversion
provenance.

ForceVLA owns strict dimensional validation, camera-slot mapping,
unused-camera masking, image preprocessing, normalization, prompt
tokenization, future 50-step action loading, internal action padding,
and semantic 7D output slicing.
