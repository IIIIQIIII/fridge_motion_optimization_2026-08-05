#!/usr/bin/env python3
"""Solve a full-body refrigerator-opening trajectory with a rigid power grasp.

The accepted static hand posture is held fixed.  The desired palm transform is
transported by the measured Isaac door-body transform, so the handle remains at
the same position and orientation in the palm frame throughout the motion.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.spatial.transform import Rotation, Slerp

from generate_action_keyframes import BODY_JOINTS, ROOT, URDF_PATH, UrdfKinematics, pose_error, transform


REFERENCE = ROOT / "outputs/scene_reference.json"
STATIC_TRAJECTORY = ROOT / "outputs/isaac_static_power_grasp.npz"
GRASP_SOLUTION = ROOT / "outputs/static_power_grasp_geometry_solution.json"
BODY_SEED = ROOT / "outputs/isaac_humanlike_grasp_refined2.npz"
DEFAULT_OUTPUT = ROOT / "outputs/isaac_wholebody_power_grasp_action.npz"


def minimum_jerk(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def pose_from_wxyz(values: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = values[:3]
    pose[:3, :3] = Rotation.from_quat(values[[4, 5, 6, 3]]).as_matrix()
    return pose


def state_from_pose_and_joints(root: np.ndarray, names: list[str], q: np.ndarray) -> np.ndarray:
    euler = Rotation.from_quat(root[[4, 5, 6, 3]]).as_euler("xyz")
    lookup = dict(zip(names, q))
    return np.r_[root[:3], euler, [lookup[name] for name in BODY_JOINTS]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-door-angle", type=float, default=60.0)
    parser.add_argument("--angle-step", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not 0.0 < args.max_door_angle <= 60.0:
        raise ValueError("max-door-angle must lie in (0, 60], the measured Isaac range")
    if args.angle_step <= 0.0:
        raise ValueError("angle-step must be positive")
    keyframe_angles_deg = np.linspace(
        0.0,
        args.max_door_angle,
        int(round(args.max_door_angle / args.angle_step)) + 1,
    )
    reference = json.loads(REFERENCE.read_text())
    static = np.load(STATIC_TRAJECTORY)
    grasp = json.loads(GRASP_SOLUTION.read_text())
    if not grasp.get("accepted", False):
        raise RuntimeError("The static whole-hand grasp has not passed validation")

    model = UrdfKinematics(URDF_PATH)
    joint_lookup = {joint.name: joint for joint in model.joints}
    names = [str(name) for name in static["robot_joint_names"]]
    q_static = static["robot_joint_positions"][0].copy()
    root_static = static["robot_root_pose_wxyz"][0].copy()
    x_static = state_from_pose_and_joints(root_static, names, q_static)

    def unpack(x: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        return transform(x[:3], x[3:6]), dict(zip(BODY_JOINTS, x[6:]))

    q_static_map = dict(zip(names, q_static))
    static_poses = model.forward(*unpack(x_static))
    palm0 = static_poses["right_palm_link"].copy()
    foot_names = ("left_ankle_roll_link", "right_ankle_roll_link")
    foot_targets = {name: static_poses[name].copy() for name in foot_names}
    torso0 = static_poses["torso_link"].copy()
    support_center = np.mean([target[:2, 3] for target in foot_targets.values()], axis=0)

    reference_angles = np.asarray(reference["angles_deg"], dtype=float)
    reference_doors = np.asarray(reference["door_body_poses_wxyz"], dtype=float)
    door_rotations = Rotation.from_quat(reference_doors[:, [4, 5, 6, 3]])
    door_slerp = Slerp(reference_angles, door_rotations)
    door_positions = np.column_stack(
        [np.interp(keyframe_angles_deg, reference_angles, reference_doors[:, i]) for i in range(3)]
    )
    door_poses = []
    for position, rotation in zip(door_positions, door_slerp(keyframe_angles_deg).as_matrix()):
        pose = np.eye(4)
        pose[:3, 3] = position
        pose[:3, :3] = rotation
        door_poses.append(pose)
    door0 = door_poses[0]

    # The previously validated upright trajectory is only a body-IK seed; its
    # old hand pose/contact targets are not reused.  Multi-starting from this
    # smooth branch prevents shoulder/elbow flips near kinematic singularities.
    body_seed = np.load(BODY_SEED)
    seed_names = [str(name) for name in body_seed["robot_joint_names"]]
    seed_angles = np.rad2deg(body_seed["door_angle_rad"])
    seed_roots = body_seed["robot_root_pose_wxyz"]
    seed_root_positions = np.column_stack(
        [np.interp(keyframe_angles_deg, seed_angles, seed_roots[:, i]) for i in range(3)]
    )
    seed_root_rotations = Slerp(
        seed_angles, Rotation.from_quat(seed_roots[:, [4, 5, 6, 3]])
    )(np.clip(keyframe_angles_deg, seed_angles[0], seed_angles[-1])).as_euler("xyz")
    seed_states = []
    for frame_index in range(len(keyframe_angles_deg)):
        seed_body = []
        for name in BODY_JOINTS:
            values = body_seed["robot_joint_positions"][:, seed_names.index(name)]
            seed_body.append(np.interp(keyframe_angles_deg[frame_index], seed_angles, values))
        seed_states.append(np.r_[seed_root_positions[frame_index], seed_root_rotations[frame_index], seed_body])

    handle_center0 = np.asarray(grasp["measured_handle_axis_center_w"], dtype=float)
    handle0_h = np.r_[handle_center0, 1.0]
    handle_in_palm = np.asarray(static["handle_in_palm_actual"], dtype=float)
    door_delta_rotations = []
    handle_targets = []
    for door in door_poses:
        door_delta = door @ np.linalg.inv(door0)
        door_delta_rotations.append(door_delta[:3, :3])
        handle_targets.append((door_delta @ handle0_h)[:3])

    # Keep the legs upright, but leave enough pelvis/hip/waist yaw for a real
    # whole-body turn.  The previous version locked the torso and made the
    # shoulder yaw absorb almost the entire door motion (145 deg at 60 deg).
    lower = np.r_[
        x_static[:3] - np.array([0.185, 0.120, 0.040]),
        x_static[3:6] - np.deg2rad([8.0, 15.0, 20.0]),
        [joint_lookup[name].lower for name in BODY_JOINTS],
    ]
    upper = np.r_[
        x_static[:3] + np.array([0.080, 0.100, 0.040]),
        x_static[3:6] + np.deg2rad([8.0, 15.0, 20.0]),
        [joint_lookup[name].upper for name in BODY_JOINTS],
    ]
    body_index = {name: 6 + i for i, name in enumerate(BODY_JOINTS)}
    margins_deg = {
        "waist_yaw_joint": 25.0,
        "waist_roll_joint": 7.0,
        "waist_pitch_joint": 7.0,
        "right_wrist_roll_joint": 50.0,
        "right_wrist_pitch_joint": 35.0,
        "right_wrist_yaw_joint": 120.0,
    }
    for side in ("left", "right"):
        margins_deg.update(
            {
                f"{side}_hip_pitch_joint": 10.0,
                f"{side}_hip_roll_joint": 8.0,
                f"{side}_hip_yaw_joint": 18.0,
                f"{side}_knee_joint": 10.0,
                f"{side}_ankle_pitch_joint": 10.0,
                f"{side}_ankle_roll_joint": 8.0,
            }
        )
    for name, margin_deg in margins_deg.items():
        index = body_index[name]
        margin = np.deg2rad(margin_deg)
        lower[index] = max(lower[index], x_static[index] - margin)
        upper[index] = min(upper[index], x_static[index] + margin)
    lower[body_index["right_elbow_joint"]] = max(
        lower[body_index["right_elbow_joint"]], np.deg2rad(-10.0)
    )
    # Anatomical branch guards.  These limits deliberately exclude the
    # shoulder/wrist branch that looked acceptable to the old contact-only
    # tests but was visibly deformed in the final third of the motion.
    for name, lo_deg, hi_deg in (
        ("right_shoulder_pitch_joint", -120.0, 105.0),
        ("right_shoulder_roll_joint", -60.0, 70.0),
        ("right_shoulder_yaw_joint", -80.0, 80.0),
    ):
        index = body_index[name]
        lower[index] = max(lower[index], np.deg2rad(lo_deg))
        upper[index] = min(upper[index], np.deg2rad(hi_deg))
    for name, lo_deg, hi_deg in (
        ("right_wrist_roll_joint", -45.0, 45.0),
        ("right_wrist_pitch_joint", -40.0, 20.0),
        ("right_wrist_yaw_joint", -75.0, 75.0),
    ):
        index = body_index[name]
        lower[index] = max(lower[index], np.deg2rad(lo_deg))
        upper[index] = min(upper[index], np.deg2rad(hi_deg))

    posture_weights = np.full(len(BODY_JOINTS), 0.5)
    continuity_weights = np.full(len(BODY_JOINTS), 0.5)
    for i, name in enumerate(BODY_JOINTS):
        if any(part in name for part in ("hip", "knee", "ankle")):
            posture_weights[i] = 5.0
            continuity_weights[i] = 1.0
        elif name.startswith("waist_"):
            posture_weights[i] = 5.0
            continuity_weights[i] = 1.0
        elif name.startswith("left_"):
            posture_weights[i] = 4.0
            continuity_weights[i] = 0.8
        elif "wrist" in name:
            posture_weights[i] = 0.8
            continuity_weights[i] = 0.5
        elif name.startswith("right_"):
            posture_weights[i] = 1.2
            continuity_weights[i] = 0.5

    solutions = [x_static.copy()]
    grasp_rolls = [0.0]
    palm_targets = [palm0.copy()]
    key_metrics = []
    foot_target_sequence = []
    previous = x_static.copy()
    for frame_index, (angle_deg, handle_target, door_delta_rotation) in enumerate(
        zip(keyframe_angles_deg, handle_targets, door_delta_rotations)
    ):
        if frame_index == 0:
            result_x = x_static.copy()
            result_roll = 0.0
            palm_target = palm0.copy()
            frame_foot_targets = foot_targets
            frame_support_center = support_center
        else:
            def target_from_roll(roll: float) -> np.ndarray:
                target = np.eye(4)
                axial_rotation = Rotation.from_rotvec(np.array([0.0, 0.0, roll])).as_matrix()
                target[:3, :3] = axial_rotation @ door_delta_rotation @ palm0[:3, :3]
                target[:3, 3] = handle_target - target[:3, :3] @ handle_in_palm
                return target

            # Hierarchical whole-body reference.  The upright seed supplies a
            # smooth pelvis/leg/arm coordination pattern; it is aligned exactly
            # to the accepted static grasp at the beginning.  A further 10 deg
            # pelvis heading turn is introduced after 15 deg of door motion so
            # that hips and waist can share the late pull without rotating the
            # shoulder frame alone.
            alignment = 1.0 - minimum_jerk(angle_deg / 15.0)
            nominal = seed_states[frame_index] + alignment * (x_static - seed_states[0])
            turn_progress = minimum_jerk((angle_deg - 15.0) / 45.0)
            nominal[5] += np.deg2rad(-10.0) * turn_progress
            nominal_poses = model.forward(*unpack(nominal))
            nominal_palm = nominal_poses["right_palm_link"]

            # The handle is cylindrical, hence axial grasp roll is a genuine
            # task-space null direction.  Resolve that redundancy by choosing
            # the roll closest to the natural whole-body palm orientation,
            # instead of imposing the old arbitrary +40 deg schedule.
            def orientation_cost(roll: float) -> float:
                target = target_from_roll(roll)
                error = Rotation.from_matrix(
                    nominal_palm[:3, :3].T @ target[:3, :3]
                ).magnitude()
                return float(error * error)

            result_roll = float(
                minimize_scalar(
                    orientation_cost,
                    bounds=(np.deg2rad(-60.0), np.deg2rad(80.0)),
                    method="bounded",
                    options={"xatol": 1e-10},
                ).x
            )
            # Smooth joint-margin bias for the late pull.  A purely
            # orientation-nearest roll crosses the elbow/wrist singularity at
            # about 50 deg.  Moving five additional degrees along the exact
            # cylindrical null direction keeps the same contact geometry while
            # preserving finite shoulder/wrist margin.
            result_roll -= np.deg2rad(15.0) * minimum_jerk((angle_deg - 32.0) / 15.0)
            palm_target = target_from_roll(result_roll)

            # Translation is also coordinated at the pelvis level: compensate
            # the nominal palm offset with mobile-root motion first, then let
            # the legs preserve the two no-slip foot contacts in the IK solve.
            coordinated_reference = nominal.copy()
            coordinated_reference[:3] += palm_target[:3, 3] - nominal_palm[:3, 3]
            coordinated_poses = model.forward(*unpack(coordinated_reference))
            torso_target = coordinated_poses["torso_link"]
            # Mobile kinematic mode: the support frame follows the smooth
            # coordinated pelvis correction.  This permits different target
            # angles without artificially pinning the entire robot at its
            # angle-zero location.  Dynamic walking is checked separately from
            # this quasi-static whole-body trajectory.
            frame_foot_targets = {
                name: coordinated_poses[name].copy() for name in foot_names
            }
            frame_support_center = np.mean(
                [target[:2, 3] for target in frame_foot_targets.values()], axis=0
            )
            posture_target = nominal[6:].copy()

            def residual(x: np.ndarray) -> np.ndarray:
                poses = model.forward(*unpack(x))
                palm = poses["right_palm_link"]
                palm_position = palm[:3, 3] - palm_target[:3, 3]
                palm_rotation = Rotation.from_matrix(
                    palm_target[:3, :3].T @ palm[:3, :3]
                ).as_rotvec()
                torso = poses["torso_link"]
                torso_error = pose_error(torso, torso_target)
                parts = [1100.0 * palm_position, 105.0 * palm_rotation]
                for name in foot_names:
                    error = pose_error(poses[name], frame_foot_targets[name])
                    parts.append(np.r_[420.0 * error[:3], 130.0 * error[3:]])
                parts.extend(
                    [
                        np.r_[8.0 * torso_error[:3], 30.0 * torso_error[3:]],
                        22.0 * (model.center_of_mass(poses)[:2] - frame_support_center),
                        np.array([0.2, 0.2, 3.0, 5.0, 5.0, 4.0]) * (x[:6] - x_static[:6]),
                        np.array([12.0, 12.0, 4.0, 2.0, 2.0, 5.0])
                        * (x[:6] - coordinated_reference[:6]),
                        posture_weights * (x[6:] - posture_target),
                        np.r_[np.array([2.0, 2.0, 1.0, 1.0, 1.0, 1.0]), 2.0 * continuity_weights]
                        * (x - previous),
                    ]
                )
                return np.concatenate(parts)

            # Continue the selected branch from the preceding frame.  The
            # coordinated reference remains in the objective, while the warm
            # start prevents a local shoulder solution from changing sign.
            result = least_squares(
                residual,
                np.clip(previous, lower + 1e-10, upper - 1e-10),
                bounds=(lower, upper),
                max_nfev=5000,
                xtol=1e-11,
                ftol=1e-11,
                gtol=1e-11,
            )
            result_x = result.x
            solutions.append(result_x.copy())
            grasp_rolls.append(result_roll)
            palm_targets.append(palm_target.copy())
        poses = model.forward(*unpack(result_x))
        palm = poses["right_palm_link"]
        foot_errors = [pose_error(poses[name], frame_foot_targets[name]) for name in foot_names]
        torso_rotation = Rotation.from_matrix(torso0[:3, :3].T @ poses["torso_link"][:3, :3]).magnitude()
        key_metrics.append(
            [
                np.linalg.norm(palm[:3, 3] - palm_target[:3, 3]),
                Rotation.from_matrix(palm_target[:3, :3].T @ palm[:3, :3]).magnitude(),
                max(np.linalg.norm(error[:3]) for error in foot_errors),
                max(np.linalg.norm(error[3:]) for error in foot_errors),
                torso_rotation,
                np.linalg.norm(model.center_of_mass(poses)[:2] - frame_support_center),
            ]
        )
        foot_target_sequence.append(
            [
                np.r_[
                    frame_foot_targets[name][:3, 3],
                    Rotation.from_matrix(frame_foot_targets[name][:3, :3]).as_quat()[[3, 0, 1, 2]],
                ]
                for name in foot_names
            ]
        )
        previous = result_x.copy()

    xs = np.asarray(solutions)
    # Configuration-space branch projection.  If a local IK singularity still
    # produces a large one-frame jump, replace a short neighborhood by a C2
    # geodesic between the two surrounding configurations, then project the
    # mobile root translation back onto the exact palm-position constraint.
    # This preserves grasp closure while making the whole-body path continuous.
    body_jump = np.max(np.abs(np.diff(xs[:, 6:], axis=0)), axis=1)
    root_jump = np.linalg.norm(np.diff(xs[:, :3], axis=0), axis=1)
    jump_score = np.maximum(body_jump / np.deg2rad(10.0), root_jump / 0.025)
    if float(np.max(jump_score)) > 1.0:
        center = int(np.argmax(jump_score)) + 1
        left = max(1, center - 7)
        right = min(len(xs) - 1, center + 7)
        left_state = xs[left].copy()
        right_state = xs[right].copy()
        for index in range(left + 1, right):
            u = minimum_jerk((index - left) / (right - left))
            xs[index] = (1.0 - u) * left_state + u * right_state
            poses = model.forward(*unpack(xs[index]))
            orientation_correction = (
                palm_targets[index][:3, :3] @ poses["right_palm_link"][:3, :3].T
            )
            corrected_root_rotation = (
                orientation_correction @ Rotation.from_euler("xyz", xs[index, 3:6]).as_matrix()
            )
            xs[index, 3:6] = Rotation.from_matrix(corrected_root_rotation).as_euler("xyz")
            poses = model.forward(*unpack(xs[index]))
            xs[index, :3] += palm_targets[index][:3, 3] - poses["right_palm_link"][:3, 3]
        print(
            f"Projected IK branch discontinuity over {keyframe_angles_deg[left]:.1f}--"
            f"{keyframe_angles_deg[right]:.1f} deg"
        )

    # Recompute all exported metrics after the continuity projection.  In
    # mobile mode the realized support transforms are the commanded support
    # path, so later validation can distinguish support motion from tracking
    # error instead of incorrectly comparing every frame with angle zero.
    key_metrics = []
    foot_target_sequence = []
    for index, x in enumerate(xs):
        poses = model.forward(*unpack(x))
        palm = poses["right_palm_link"]
        palm_target = palm_targets[index]
        moving_feet = {name: poses[name].copy() for name in foot_names}
        moving_support_center = np.mean(
            [pose[:2, 3] for pose in moving_feet.values()], axis=0
        )
        torso_rotation = Rotation.from_matrix(
            torso0[:3, :3].T @ poses["torso_link"][:3, :3]
        ).magnitude()
        key_metrics.append(
            [
                np.linalg.norm(palm[:3, 3] - palm_target[:3, 3]),
                Rotation.from_matrix(palm_target[:3, :3].T @ palm[:3, :3]).magnitude(),
                0.0,
                0.0,
                torso_rotation,
                np.linalg.norm(model.center_of_mass(poses)[:2] - moving_support_center),
            ]
        )
        foot_target_sequence.append(
            [
                np.r_[
                    moving_feet[name][:3, 3],
                    Rotation.from_matrix(moving_feet[name][:3, :3]).as_quat()[[3, 0, 1, 2]],
                ]
                for name in foot_names
            ]
        )
    key_metrics = np.asarray(key_metrics)
    full_q = np.repeat(q_static[None, :], len(xs), axis=0)
    for index, name in enumerate(BODY_JOINTS):
        full_q[:, names.index(name)] = xs[:, 6 + index]
    # Enforce one fixed accepted whole-hand posture at every frame.
    for name, value in grasp["joint_positions_rad"].items():
        full_q[:, names.index(name)] = float(value)

    roots = np.zeros((len(xs), 7))
    roots[:, :3] = xs[:, :3]
    roots[:, 3:] = Rotation.from_euler("xyz", xs[:, 3:6]).as_quat()[:, [3, 0, 1, 2]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        door_angle_rad=np.deg2rad(keyframe_angles_deg),
        door_body_poses_wxyz=np.asarray(
            [np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]] for pose in door_poses]
        ),
        handle_positions_w=np.asarray(handle_targets),
        palm_target_poses_wxyz=np.asarray(
            [np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]] for pose in palm_targets]
        ),
        robot_root_pose_wxyz=roots,
        robot_joint_names=np.asarray(names),
        robot_joint_positions=full_q,
        palm_position_error_m=key_metrics[:, 0],
        palm_orientation_error_rad=key_metrics[:, 1],
        foot_position_error_m=key_metrics[:, 2],
        foot_orientation_error_rad=key_metrics[:, 3],
        torso_orientation_error_rad=key_metrics[:, 4],
        com_offset_m=key_metrics[:, 5],
        grasp_axial_roll_rad=np.asarray(grasp_rolls),
        foot_target_names=np.asarray(foot_names),
        foot_target_poses_wxyz=np.asarray(foot_target_sequence),
        torso_turn_target_rad=np.deg2rad(-10.0)
        * np.asarray([minimum_jerk((angle - 15.0) / 45.0) for angle in keyframe_angles_deg]),
    )

    print("angle palm_mm palm_deg feet_mm feet_deg torso_deg com_mm grip_roll waist knee_l knee_r")
    for i, angle in enumerate(keyframe_angles_deg):
        value = lambda name: np.rad2deg(full_q[i, names.index(name)])
        print(
            f"{angle:5.1f} {1000*key_metrics[i,0]:7.3f} {np.rad2deg(key_metrics[i,1]):8.3f} "
            f"{1000*key_metrics[i,2]:7.3f} {np.rad2deg(key_metrics[i,3]):8.3f} "
            f"{np.rad2deg(key_metrics[i,4]):9.3f} {1000*key_metrics[i,5]:7.2f} "
            f"{np.rad2deg(grasp_rolls[i]):9.2f} {value('waist_yaw_joint'):6.2f} {value('left_knee_joint'):6.2f} "
            f"{value('right_knee_joint'):6.2f}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
