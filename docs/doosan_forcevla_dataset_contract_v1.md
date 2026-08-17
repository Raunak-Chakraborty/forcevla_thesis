# Doosan ForceVLA Dataset Contract v1

Contract ID: `doosan_forcevla_dataset_contract_v1`

Status: FROZEN baseline contract for the Doosan M1013 ForceVLA thesis pipeline.

This document defines the boundary between `doosan-forcevla-data-tools`
and the ForceVLA thesis repository. Raw acquisition format is outside
this contract. The converter may change internally as long as its final
output satisfies this interface.

## 1. Required model-facing fields

Each aligned dataset sample must provide:

- `observation.images.tcp_camera`
- `observation.images.external_camera_2`
- `observation.state`
- `action`
- `prompt`

The dataset must also preserve correct temporal and episode identity,
including `episode_index`, `frame_index`, and `timestamp`.

## 2. Production cameras

Only two physical RGB cameras are part of Contract v1.

### tcp_camera

- Hardware: Intel RealSense D405
- Serial: `409122273671`
- Physical role: robot-mounted TCP / wrist view
- Raw RGB resolution: 640 x 480
- Raw acquisition rate: approximately 30 Hz
- Image topic: `/doosan_cameras/tcp_camera/color/image_raw`
- CameraInfo topic: `/doosan_cameras/tcp_camera/color/camera_info`

### external_camera_2

- Hardware: Intel RealSense D435I
- Serial: `233622070850`
- Physical role: fixed external workspace view
- Raw RGB resolution: 640 x 480
- Raw acquisition rate: approximately 60 Hz
- Image topic: `/doosan_cameras/external_camera_2/color/image_raw`
- CameraInfo topic: `/doosan_cameras/external_camera_2/color/camera_info`

### Excluded legacy camera

`external_camera_1`, Intel RealSense D435, serial `920312073254`,
is NOT part of Contract v1 and must not be required or fabricated by
the converter.

Depth is not part of Contract v1.

## 3. Synchronization

Raw acquisition uses rosbag2 MCAP and does not perform cross-modal
synchronization online.

Offline synchronization and downsampling are owned by
`doosan-forcevla-data-tools`.

The reference visual timeline is `tcp_camera`.

The final ForceVLA dataset target rate is 30 Hz.

The approximately 60 Hz `external_camera_2` stream must be aligned or
downsampled onto the TCP-camera reference timeline.

Robot, gripper, wrench, and other state streams must be aligned onto
the same final timeline.

## 4. ForceVLA image-slot mapping

The pretrained model retains three internal image slots although the
robot uses only two real cameras.

Mapping:

- `external_camera_2` -> `base_0_rgb`, mask = true
- `tcp_camera` -> `left_wrist_0_rgb`, mask = true
- no third camera -> `right_wrist_0_rgb`, zero image, mask = false

The zero third image is created inside ForceVLA.

The converted dataset must contain only the two real camera views.

## 5. Semantic state

`observation.state` has exactly 13 float channels.

Ordering:

- 0: TCP position x
- 1: TCP position y
- 2: TCP position z
- 3: TCP rotation-vector x
- 4: TCP rotation-vector y
- 5: TCP rotation-vector z
- 6: measured gripper open fraction
- 7: force x
- 8: force y
- 9: force z
- 10: torque x
- 11: torque y
- 12: torque z

Equivalent grouping:

`[position_3, rotation_vector_3, gripper_1, wrench_6]`

Contract-v1 physical conventions:

- TCP position: absolute, robot base frame, metres
- TCP orientation: absolute SO(3) rotation vector for base-to-TCP
  orientation, radians
- gripper: normalized measured opening, 0 = closed, 1 = open
- wrench ordering: `[Fx, Fy, Fz, Tx, Ty, Tz]`
- wrench frame: TCP / tool frame
- force units: newtons
- torque units: newton-metres

Any wrench bias, tare, gravity compensation, or sensor compensation
performed upstream must be deterministic and recorded in dataset
provenance.

All state values must be finite.

## 6. Semantic action

`action` has exactly 7 float channels.

Ordering:

- 0: delta TCP x
- 1: delta TCP y
- 2: delta TCP z
- 3: delta rotation-vector x
- 4: delta rotation-vector y
- 5: delta rotation-vector z
- 6: absolute gripper target open fraction

Translation action:

`delta_p = p(t+1) - p(t)`

It is expressed in the robot base frame and measured in metres.

Rotation action uses the relative orientation:

`delta_R = R(t+1) @ R(t).T`

and stores `Log(delta_R)` as a 3D rotation vector in radians.

The gripper action is an absolute normalized target:

- 0 = fully closed
- 1 = fully open

All action values must be finite.

## 7. Critical delta-action rule

The dataset already stores Cartesian delta actions.

ForceVLA must NOT apply another `DeltaActions` operation.

The converter must NOT:

- difference the first six action channels twice
- emit joint-space deltas instead
- pad the semantic action to 32 dimensions

## 8. External versus internal dimensions

The external data contract is always:

- state: 13D
- action: 7D

The baseline checkpoint-compatible ForceVLA model may internally use a
32D action representation.

That padding is strictly an internal ForceVLA implementation detail.

`doosan-forcevla-data-tools` must always export the semantic 7D action.

## 9. Action horizon

ForceVLA action horizon is 50.

The dataset stores one semantic 7D action per aligned frame.

The LeRobot / ForceVLA loading pipeline constructs future action
sequences of 50 actions.

Episode identity must be correct so an action sequence never represents
motion from a different episode.

## 10. Prompt

The ForceVLA-facing sample must contain:

`prompt: non-empty string`

Example:

`Insert the peg into the hole.`

A source dataset may also retain `task`, but the ForceVLA-facing
contract requires the resulting `prompt`.

## 11. Normalization

The converter exports values using the physical semantic units in this
document.

ForceVLA normalization must not be permanently baked into the dataset.

Normalization statistics belong to the ForceVLA/OpenPI training
pipeline.

When train/validation/test episode holdouts are used, final
normalization statistics must be computed from training episodes only.

## 12. Converter responsibilities

`doosan-forcevla-data-tools` owns:

- MCAP decoding
- timestamp extraction
- TCP-camera reference timeline generation
- offline synchronization and resampling
- external-camera alignment
- robot-state alignment
- gripper alignment
- wrench alignment
- pose representation conversion
- 7D Cartesian action construction
- episode and frame indexing
- LeRobot export
- conversion provenance

## 13. ForceVLA responsibilities

ForceVLA owns:

- strict 13D state validation
- strict 7D semantic action validation
- required two-camera validation
- mapping the two cameras into pretrained model slots
- zero-padding and masking the unused third model slot
- image preprocessing
- normalization
- prompt tokenization
- 50-step future action consumption
- internal checkpoint-compatibility padding
- 7D semantic output slicing

## 14. Contract rejection conditions

A Contract-v1 sample is invalid if:

- `tcp_camera` is absent
- `external_camera_2` is absent
- state dimension is not exactly 13
- semantic action dimension is not exactly 7
- state or action contains NaN or infinity
- camera identity is ambiguous
- timestamp alignment is ambiguous
- pose convention is ambiguous
- physical units are ambiguous
- wrench ordering or frame is ambiguous
- episode boundaries are ambiguous
- Cartesian actions are differenced a second time

`external_camera_1` is not required.

## 15. Executable contract

The executable implementation and tests are:

- `src/openpi/policies/forcevla_policy.py`
- `src/openpi/training/config.py`
- `src/openpi/policies/doosan_forcevla_contract_test.py`
- `src/openpi/models/pi0_force_contract_test.py`
- `src/openpi/models/pi0_force_runtime_contract_test.py`

This document and those executable tests must remain consistent.

Any future incompatible change to camera requirements, state ordering,
action ordering, units, frames, rotation convention, gripper
convention, or action semantics requires a new contract version.
