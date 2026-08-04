#!/usr/bin/env python3
"""Solve balanced, orientation-aware G1 fridge-opening keyframes.

Compared with the first Isaac-aligned solve, this version keeps the torso near
its pre-grasp attitude, strongly centers the approximate COM over the feet,
tracks a full 6D palm pose, and closes the Wuji right hand around the handle.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from generate_action_keyframes import BODY_JOINTS, ROOT, URDF_PATH, UrdfKinematics, pose_error, transform


REFERENCE = ROOT / "outputs/scene_reference.json"
OUTPUT = ROOT / "outputs/isaac_balanced_grasp_action.npz"
RIGHT_FINGER_JOINTS = [f"right_finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5)]
RIGHT_FINGER_TIPS = [f"right_finger{finger}_tip_link" for finger in range(1, 6)]

# A cylindrical power-grasp posture for the Wuji hand.  Finger 1 is treated as
# the opposing digit; fingers 2--5 wrap with progressively similar flexion.
RIGHT_HAND_GRASP = {
    **{f"right_finger1_joint{joint}": value for joint, value in enumerate((0.75, 0.15, 0.90, 0.90), 1)},
    **{
        f"right_finger{finger}_joint{joint}": value
        for finger in range(2, 6)
        for joint, value in enumerate((0.65, 0.00, 0.95, 0.95), 1)
    },
}
LEFT_HAND_RELAXED = {
    **{f"left_finger1_joint{joint}": value for joint, value in enumerate((0.20, 0.05, 0.25, 0.25), 1)},
    **{
        f"left_finger{finger}_joint{joint}": value
        for finger in range(2, 6)
        for joint, value in enumerate((0.15, 0.00, 0.20, 0.20), 1)
    },
}


def pose_from_wxyz(values: list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.eye(4)
    out[:3, 3] = values[:3]
    out[:3, :3] = Rotation.from_quat(values[[4, 5, 6, 3]]).as_matrix()
    return out


def main():
    reference = json.loads(REFERENCE.read_text())
    model = UrdfKinematics(URDF_PATH)
    joint_lookup = {joint.name: joint for joint in model.joints}
    sim_names = list(reference["robot_joint_names"])
    sim_q0 = np.asarray(reference["robot_joint_positions"], dtype=float)
    q0_by_name = dict(zip(sim_names, sim_q0))
    q0 = {name: q0_by_name[name] for name in BODY_JOINTS}

    root_pose = np.asarray(reference["robot_root_pose_wxyz"], dtype=float)
    rpy0 = Rotation.from_quat(root_pose[[4, 5, 6, 3]]).as_euler("xyz")
    x0 = np.r_[root_pose[:3], rpy0, [q0[name] for name in BODY_JOINTS]]

    def unpack(x):
        return transform(x[:3], x[3:6]), dict(zip(BODY_JOINTS, x[6:]))

    poses0 = model.forward(*unpack(x0))
    foot_names = ("left_ankle_roll_link", "right_ankle_roll_link")
    foot_targets = {name: poses0[name].copy() for name in foot_names}
    support_center = np.mean([foot_targets[name][:2, 3] for name in foot_names], axis=0)
    torso_target = poses0["torso_link"].copy()
    palm_R0 = poses0["right_palm_link"][:3, :3].copy()

    # Root translation and roll/pitch are deliberately tighter than v1.  Yaw
    # remains moderately free so the robot can follow the door arc.
    root_margin = np.deg2rad([10.0, 8.0, 22.0])
    lower = np.r_[
        x0[:3] - np.array([0.10, 0.10, 0.08]),
        x0[3:6] - root_margin,
        [joint_lookup[name].lower for name in BODY_JOINTS],
    ]
    upper = np.r_[
        x0[:3] + np.array([0.10, 0.10, 0.08]),
        x0[3:6] + root_margin,
        [joint_lookup[name].upper for name in BODY_JOINTS],
    ]

    # Prevent the optimizer from exchanging an arm motion for a large waist
    # bend.  Keep waist pitch/roll within a compact neighborhood of pre-grasp.
    body_index = {name: 6 + index for index, name in enumerate(BODY_JOINTS)}
    for name, margin_deg in (("waist_pitch_joint", 10.0), ("waist_roll_joint", 12.0), ("waist_yaw_joint", 32.0)):
        index = body_index[name]
        margin = np.deg2rad(margin_deg)
        lower[index] = max(lower[index], x0[index] - margin)
        upper[index] = min(upper[index], x0[index] + margin)
    for name, margin_deg in (
        ("right_wrist_roll_joint", 22.0),
        ("right_wrist_pitch_joint", 22.0),
        ("right_wrist_yaw_joint", 18.0),
    ):
        index = body_index[name]
        margin = np.deg2rad(margin_deg)
        lower[index] = max(lower[index], x0[index] - margin)
        upper[index] = min(upper[index], x0[index] + margin)

    posture_weights = np.full(len(BODY_JOINTS), 0.85)
    posture_target = x0[6:].copy()
    neutral_left_arm = {
        "left_shoulder_pitch_joint": 0.20,
        "left_shoulder_roll_joint": 0.20,
        "left_shoulder_yaw_joint": 0.00,
        "left_elbow_joint": 0.60,
        "left_wrist_roll_joint": 0.00,
        "left_wrist_pitch_joint": 0.00,
        "left_wrist_yaw_joint": 0.00,
    }
    for index, name in enumerate(BODY_JOINTS):
        if name.startswith("right_") and any(part in name for part in ("shoulder", "elbow", "wrist")):
            posture_weights[index] = 0.10
            if "wrist" in name:
                posture_weights[index] = 2.8
        elif name.startswith("left_") and any(part in name for part in ("shoulder", "elbow", "wrist")):
            posture_weights[index] = 2.4
            posture_target[index] = neutral_left_arm[name]
        elif name in ("waist_pitch_joint", "waist_roll_joint"):
            posture_weights[index] = 3.2
        elif name == "waist_yaw_joint":
            posture_weights[index] = 0.65

    handle_positions = np.asarray(reference["handle_positions_w"], dtype=float)
    door_poses = [pose_from_wxyz(values) for values in reference["door_body_poses_wxyz"]]
    door_R0 = door_poses[0][:3, :3]
    angles = np.deg2rad(reference["angles_deg"])

    # In the palm frame, the curled fingertips cluster near x=0.06, z=0.10.
    # Put the handle there instead of placing the palm origin inside the handle.
    handle_in_palm = np.array([0.060, 0.000, 0.105])

    solutions, palm_errors, palm_orientation_errors = [], [], []
    foot_errors, com_offsets, torso_tilts = [], [], []
    previous = x0.copy()

    for frame_index, handle in enumerate(handle_positions):
        palm_target = np.eye(4)
        # A vertical cylindrical handle does not require the hand to rotate
        # rigidly with the door panel.  Preserve the comfortable pre-grasp
        # world orientation and let shoulder/elbow motion follow the arc.
        palm_target[:3, :3] = palm_R0

        def residual(x):
            poses = model.forward(*unpack(x))
            palm = poses["right_palm_link"]
            grasp_center_err = palm[:3, 3] + palm[:3, :3] @ handle_in_palm - handle
            palm_orientation_err = Rotation.from_matrix(palm_R0.T @ palm[:3, :3]).as_rotvec()
            torso_err = pose_error(poses["torso_link"], torso_target)
            com = model.center_of_mass(poses)
            parts = [np.r_[85.0 * grasp_center_err, 2.0 * palm_orientation_err]]
            for foot_name in foot_names:
                err = pose_error(poses[foot_name], foot_targets[foot_name])
                parts.append(np.r_[95.0 * err[:3], 42.0 * err[3:]])
            parts.append(12.0 * (com[:2] - support_center))
            parts.append(np.r_[18.0 * torso_err[:2], 1.0 * torso_err[2]])
            parts.append(np.array([3.5, 3.5, 2.0, 3.0, 3.0, 1.2]) * (x[:6] - x0[:6]))
            parts.append(posture_weights * (x[6:] - posture_target))
            if frame_index:
                parts.append(np.r_[np.ones(6) * 1.5, np.ones(len(BODY_JOINTS)) * 0.35] * (x - previous))
            return np.concatenate(parts)

        result = least_squares(
            residual,
            previous,
            bounds=(lower, upper),
            max_nfev=2400,
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
        )
        poses = model.forward(*unpack(result.x))
        palm = poses["right_palm_link"]
        grasp_center_err = palm[:3, 3] + palm[:3, :3] @ handle_in_palm - handle
        palm_orientation_err = Rotation.from_matrix(palm_R0.T @ palm[:3, :3]).as_rotvec()
        torso_relative = torso_target[:3, :3].T @ poses["torso_link"][:3, :3]
        solutions.append(result.x.copy())
        palm_errors.append(np.linalg.norm(grasp_center_err))
        palm_orientation_errors.append(np.linalg.norm(palm_orientation_err))
        foot_errors.append(max(np.linalg.norm(poses[name][:3, 3] - foot_targets[name][:3, 3]) for name in foot_names))
        com_offsets.append(np.linalg.norm(model.center_of_mass(poses)[:2] - support_center))
        torso_tilts.append(np.linalg.norm(Rotation.from_matrix(torso_relative).as_rotvec()[:2]))
        previous = result.x.copy()

    xs = np.vstack(solutions)
    full_joint_q = np.repeat(sim_q0[None, :], len(xs), axis=0)
    for index, name in enumerate(BODY_JOINTS):
        full_joint_q[:, sim_names.index(name)] = xs[:, 6 + index]
    for name, value in LEFT_HAND_RELAXED.items():
        full_joint_q[:, sim_names.index(name)] = value

    root_poses = np.zeros((len(xs), 7))
    root_poses[:, :3] = xs[:, :3]
    xyzw = Rotation.from_euler("xyz", xs[:, 3:6]).as_quat()
    root_poses[:, 3:] = xyzw[:, [3, 0, 1, 2]]

    # Solve the 20 right-hand joints against explicit, finite contact points:
    # thumb on the palm-side surface and the other four fingers on the opposite
    # surface at distinct heights along the handle.
    finger_joint_lookup = {name: joint_lookup[name] for name in RIGHT_FINGER_JOINTS}
    finger_lower = np.asarray([finger_joint_lookup[name].lower for name in RIGHT_FINGER_JOINTS])
    finger_upper = np.asarray([finger_joint_lookup[name].upper for name in RIGHT_FINGER_JOINTS])
    grasp_seed = np.asarray([RIGHT_HAND_GRASP[name] for name in RIGHT_FINGER_JOINTS])
    fingertip_contact_errors = []
    previous_fingers = grasp_seed.copy()
    for frame_index, x in enumerate(xs):
        q_body = dict(zip(sim_names, full_joint_q[frame_index]))
        body_poses = model.forward(transform(x[:3], x[3:6]), q_body)
        handle = handle_positions[frame_index]
        palm_pos = body_poses["right_palm_link"][:3, 3]
        approach = handle - palm_pos
        approach[2] = 0.0
        approach /= max(np.linalg.norm(approach), 1e-9)
        handle_radius = 0.012
        contact_targets = np.vstack(
            [
                handle - handle_radius * approach + np.array([0.0, 0.0, 0.045]),
                handle + handle_radius * approach + np.array([0.0, 0.0, 0.060]),
                handle + handle_radius * approach + np.array([0.0, 0.0, 0.020]),
                handle + handle_radius * approach + np.array([0.0, 0.0, -0.020]),
                handle + handle_radius * approach + np.array([0.0, 0.0, -0.060]),
            ]
        )

        def finger_residual(q_fingers):
            q_all = dict(q_body)
            q_all.update(zip(RIGHT_FINGER_JOINTS, q_fingers))
            poses = model.forward(transform(x[:3], x[3:6]), q_all)
            tips = np.vstack([poses[name][:3, 3] for name in RIGHT_FINGER_TIPS])
            parts = [120.0 * (tips - contact_targets).ravel(), 0.22 * (q_fingers - grasp_seed)]
            if frame_index:
                parts.append(0.10 * (q_fingers - previous_fingers))
            return np.concatenate(parts)

        finger_result = least_squares(
            finger_residual,
            previous_fingers,
            bounds=(finger_lower, finger_upper),
            max_nfev=1600,
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
        )
        for index, name in enumerate(RIGHT_FINGER_JOINTS):
            full_joint_q[frame_index, sim_names.index(name)] = finger_result.x[index]
        q_all = dict(zip(sim_names, full_joint_q[frame_index]))
        poses = model.forward(transform(x[:3], x[3:6]), q_all)
        tips = np.vstack([poses[name][:3, 3] for name in RIGHT_FINGER_TIPS])
        fingertip_contact_errors.append(np.linalg.norm(tips - contact_targets, axis=1))
        previous_fingers = finger_result.x.copy()

    np.savez(
        OUTPUT,
        door_angle_rad=angles,
        handle_positions_w=handle_positions,
        robot_root_pose_wxyz=root_poses,
        robot_joint_names=np.asarray(sim_names),
        robot_joint_positions=full_joint_q,
        palm_error_m=np.asarray(palm_errors),
        palm_orientation_error_rad=np.asarray(palm_orientation_errors),
        foot_error_m=np.asarray(foot_errors),
        com_offset_m=np.asarray(com_offsets),
        torso_tilt_error_rad=np.asarray(torso_tilts),
        fingertip_contact_error_m=np.asarray(fingertip_contact_errors),
        grasp_joint_names=np.asarray(RIGHT_FINGER_JOINTS),
    )

    print("angle palm_mm palm_deg foot_mm com_mm torso_tilt_deg fingertip_contact_mean_mm")
    for index, angle in enumerate(np.rad2deg(angles)):
        print(
            f"{angle:5.1f} {1000*palm_errors[index]:7.2f} "
            f"{np.rad2deg(palm_orientation_errors[index]):8.2f} {1000*foot_errors[index]:7.2f} "
            f"{1000*com_offsets[index]:6.1f} {np.rad2deg(torso_tilts[index]):14.2f} "
            f"{1000*np.mean(fingertip_contact_errors[index]):27.1f}"
        )
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
