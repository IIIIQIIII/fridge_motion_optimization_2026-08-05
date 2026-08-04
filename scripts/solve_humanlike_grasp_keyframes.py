#!/usr/bin/env python3
"""Generate upright, torso-stable keyframes around the real upper handle.

The earlier prototype optimized against the task command point, which is the
lowest corner of the upper-handle bounds.  This solver instead tracks the
queried mesh center and transports a full grasp frame with the door.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from generate_action_keyframes import BODY_JOINTS, ROOT, URDF_PATH, UrdfKinematics, pose_error, transform


REFERENCE = ROOT / "outputs/scene_reference.json"
GEOMETRY = ROOT / "outputs/fridge_handle_geometry.json"
OUTPUT = ROOT / "outputs/isaac_humanlike_grasp_action.npz"

RIGHT_HAND_GRASP = {
    **{f"right_finger1_joint{j}": v for j, v in enumerate((0.6979, -0.1387, 1.5623, 0.7587), 1)},
    **{f"right_finger2_joint{j}": v for j, v in enumerate((0.9907, 0.0764, 0.5865, 0.6327), 1)},
    **{f"right_finger3_joint{j}": v for j, v in enumerate((1.1013, 0.0250, 0.4591, 0.5109), 1)},
    **{f"right_finger4_joint{j}": v for j, v in enumerate((1.0113, 0.0297, 0.4939, 0.5544), 1)},
    **{f"right_finger5_joint{j}": v for j, v in enumerate((0.8825, 0.0484, 0.5146, 0.5937), 1)},
}
LEFT_HAND_RELAXED = {
    **{f"left_finger1_joint{j}": v for j, v in enumerate((0.20, 0.05, 0.25, 0.25), 1)},
    **{
        f"left_finger{finger}_joint{j}": v
        for finger in range(2, 6)
        for j, v in enumerate((0.15, 0.00, 0.20, 0.20), 1)
    },
}


def pose_from_wxyz(values: list[float]) -> np.ndarray:
    out = np.eye(4)
    out[:3, 3] = values[:3]
    w, x, y, z = values[3:]
    out[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    return out


def main() -> None:
    reference = json.loads(REFERENCE.read_text())
    geometry = json.loads(GEOMETRY.read_text())
    upper = next(item for item in geometry if item["path"].endswith("/E_handle_1_7") and item["type"] == "Xform")
    handle_center0 = np.asarray(upper["center_w"], dtype=float)
    handle_extent = np.asarray(upper["extent_w"], dtype=float)

    model = UrdfKinematics(URDF_PATH)
    joint_lookup = {joint.name: joint for joint in model.joints}
    sim_names = list(reference["robot_joint_names"])
    sim_q0 = np.asarray(reference["robot_joint_positions"], dtype=float)
    q_reference = dict(zip(sim_names, sim_q0))

    root_ref = np.asarray(reference["robot_root_pose_wxyz"], dtype=float)
    root_ref_tf = pose_from_wxyz(root_ref.tolist())
    ref_poses = model.forward(root_ref_tf, q_reference)
    reference_foot_z = np.mean(
        [ref_poses[name][2, 3] for name in ("left_ankle_roll_link", "right_ankle_roll_link")]
    )

    # Upright stance: about 17 degrees of knee flexion instead of 65--83.
    posture = {name: q_reference[name] for name in BODY_JOINTS}
    for side in ("left", "right"):
        posture[f"{side}_hip_pitch_joint"] = -0.14
        posture[f"{side}_hip_roll_joint"] = 0.0
        posture[f"{side}_hip_yaw_joint"] = 0.0
        posture[f"{side}_knee_joint"] = 0.30
        posture[f"{side}_ankle_pitch_joint"] = -0.16
        posture[f"{side}_ankle_roll_joint"] = 0.0
    for name in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"):
        posture[name] = 0.0
    posture.update(
        {
            "left_shoulder_pitch_joint": 0.18,
            "left_shoulder_roll_joint": 0.16,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.55,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        }
    )

    root_yaw = np.deg2rad(12.0)
    root_xyz = root_ref[:3].copy()
    root_xyz[:2] = np.array([0.79, 0.06])
    root_xyz[2] = 0.82
    stand_tf = transform(root_xyz, (0.0, 0.0, root_yaw))
    stand_poses = model.forward(stand_tf, posture)
    root_xyz[2] += reference_foot_z - np.mean(
        [stand_poses[name][2, 3] for name in ("left_ankle_roll_link", "right_ankle_roll_link")]
    )
    x_stand = np.r_[root_xyz, [0.0, 0.0, root_yaw], [posture[name] for name in BODY_JOINTS]]

    def unpack(x: np.ndarray):
        return transform(x[:3], x[3:6]), dict(zip(BODY_JOINTS, x[6:]))

    stand_poses = model.forward(*unpack(x_stand))
    foot_names = ("left_ankle_roll_link", "right_ankle_roll_link")
    foot_targets = {name: stand_poses[name].copy() for name in foot_names}
    torso_target = stand_poses["torso_link"].copy()
    support_center = np.mean([foot_targets[name][:2, 3] for name in foot_names], axis=0)

    lower = np.r_[
        x_stand[:3] - np.array([0.08, 0.08, 0.035]),
        x_stand[3:6] - np.deg2rad([5.0, 5.0, 10.0]),
        [joint_lookup[name].lower for name in BODY_JOINTS],
    ]
    upper_bound = np.r_[
        x_stand[:3] + np.array([0.08, 0.08, 0.035]),
        x_stand[3:6] + np.deg2rad([5.0, 5.0, 10.0]),
        [joint_lookup[name].upper for name in BODY_JOINTS],
    ]
    body_index = {name: 6 + i for i, name in enumerate(BODY_JOINTS)}

    margins = {
        "waist_yaw_joint": 12.0,
        "waist_roll_joint": 8.0,
        "waist_pitch_joint": 8.0,
        "right_wrist_roll_joint": 22.0,
        "right_wrist_pitch_joint": 22.0,
        "right_wrist_yaw_joint": 120.0,
    }
    for side in ("left", "right"):
        margins.update(
            {
                f"{side}_hip_pitch_joint": 10.0,
                f"{side}_hip_roll_joint": 8.0,
                f"{side}_hip_yaw_joint": 8.0,
                f"{side}_knee_joint": 10.0,
                f"{side}_ankle_pitch_joint": 10.0,
                f"{side}_ankle_roll_joint": 8.0,
            }
        )
    for name, margin_deg in margins.items():
        i = body_index[name]
        margin = np.deg2rad(margin_deg)
        lower[i] = max(lower[i], x_stand[i] - margin)
        upper_bound[i] = min(upper_bound[i], x_stand[i] + margin)
    # A negative elbow angle visibly hyperextends the arm in the late frames.
    lower[body_index["right_elbow_joint"]] = max(
        lower[body_index["right_elbow_joint"]], np.deg2rad(-10.0)
    )

    weights = np.full(len(BODY_JOINTS), 0.8)
    for i, name in enumerate(BODY_JOINTS):
        if any(part in name for part in ("hip", "knee", "ankle")):
            weights[i] = 5.0
        elif name.startswith("waist_"):
            weights[i] = 5.5
        elif name.startswith("left_"):
            weights[i] = 3.0
        elif "wrist" in name:
            weights[i] = 2.5
        elif name.startswith("right_"):
            weights[i] = 0.15

    door_poses = [pose_from_wxyz(v) for v in reference["door_body_poses_wxyz"]]
    door0 = door_poses[0]
    handle_local = door0[:3, :3].T @ (handle_center0 - door0[:3, 3])
    handle_centers = np.vstack([door[:3, 3] + door[:3, :3] @ handle_local for door in door_poses])

    # Palm frame convention for this hand: +x reaches toward the handle and
    # +y runs along the vertical handle.  Keep the grasp frame attached to the
    # rotating door so the handle cannot slide through the fingers.
    approach_yaw0 = np.deg2rad(60.0)
    palm_x0 = np.array([np.cos(approach_yaw0), np.sin(approach_yaw0), 0.0])
    palm_y0 = np.array([0.0, 0.0, 1.0])
    palm_z0 = np.cross(palm_x0, palm_y0)
    palm_R0 = np.column_stack((palm_x0, palm_y0, palm_z0))
    # The closed-hand FK places the five fingertips around x=0.05--0.06 and
    # z=0.08--0.10 in the palm frame.  Put the handle axis through that curl,
    # not in front of or below the fist.
    handle_in_palm = np.array([0.068, 0.000, 0.090])

    solutions = []
    metrics = []
    previous = x_stand.copy()
    for frame_index, (handle, door) in enumerate(zip(handle_centers, door_poses)):
        door_delta = door[:3, :3] @ door0[:3, :3].T
        # The bar is vertical, so the hand may roll around its axis while the
        # door swings.  Following roughly half the door yaw preserves the grip
        # without forcing the elbow into reverse extension late in the pull.
        grasp_follow = 1.0
        delta_rotvec = Rotation.from_matrix(door_delta).as_rotvec()
        target_R = Rotation.from_rotvec(grasp_follow * delta_rotvec).as_matrix() @ palm_R0
        target_p = handle - target_R @ handle_in_palm

        def residual(x: np.ndarray) -> np.ndarray:
            poses = model.forward(*unpack(x))
            palm = poses["right_palm_link"]
            torso = poses["torso_link"]
            palm_pos_err = palm[:3, 3] - target_p
            palm_rot_err = Rotation.from_matrix(target_R.T @ palm[:3, :3]).as_rotvec()
            torso_rot_err = Rotation.from_matrix(torso_target[:3, :3].T @ torso[:3, :3]).as_rotvec()
            torso_z_err = torso[2, 3] - torso_target[2, 3]
            parts = [85.0 * palm_pos_err, 6.0 * palm_rot_err]
            for name in foot_names:
                err = pose_error(poses[name], foot_targets[name])
                parts.append(np.r_[110.0 * err[:3], 35.0 * err[3:]])
            parts.extend(
                [
                    12.0 * torso_rot_err,
                    np.array([18.0 * torso_z_err]),
                    15.0 * (model.center_of_mass(poses)[:2] - support_center),
                    np.array([2.0, 2.0, 2.5, 4.0, 4.0, 3.5]) * (x[:6] - x_stand[:6]),
                    weights * (x[6:] - x_stand[6:]),
                ]
            )
            if frame_index:
                parts.append(np.r_[np.ones(6) * 0.15, np.ones(len(BODY_JOINTS)) * 0.05] * (x - previous))
            return np.concatenate(parts)

        result = least_squares(
            residual,
            x_stand,
            bounds=(lower, upper_bound),
            max_nfev=3000,
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
        )
        poses = model.forward(*unpack(result.x))
        palm = poses["right_palm_link"]
        torso = poses["torso_link"]
        metrics.append(
            [
                np.linalg.norm(palm[:3, 3] - target_p),
                np.linalg.norm(Rotation.from_matrix(target_R.T @ palm[:3, :3]).as_rotvec()),
                np.linalg.norm(Rotation.from_matrix(torso_target[:3, :3].T @ torso[:3, :3]).as_rotvec()),
                np.linalg.norm(model.center_of_mass(poses)[:2] - support_center),
            ]
        )
        solutions.append(result.x.copy())
        previous = result.x.copy()

    xs = np.vstack(solutions)
    full_q = np.repeat(sim_q0[None, :], len(xs), axis=0)
    for i, name in enumerate(BODY_JOINTS):
        full_q[:, sim_names.index(name)] = xs[:, 6 + i]
    for name, value in LEFT_HAND_RELAXED.items():
        full_q[:, sim_names.index(name)] = value
    # Refine every finger against the actual handle position in the achieved
    # palm frame.  The thumb targets the palm-side surface and the four fingers
    # target the opposite surface; this prevents the old all-fingertips-to-one-
    # point penetration while retaining a compact power-grasp seed.
    finger_surface_errors = np.zeros((len(xs), 5))
    finger_y_offsets = np.array([0.001, 0.017, 0.010, -0.009, -0.022])
    previous_fingers = {name: value for name, value in RIGHT_HAND_GRASP.items()}
    for frame_index, x in enumerate(xs):
        q_body = dict(zip(sim_names, full_q[frame_index]))
        q_body.update(previous_fingers)
        poses = model.forward(transform(x[:3], x[3:6]), q_body)
        palm = poses["right_palm_link"]
        center_palm = palm[:3, :3].T @ (handle_centers[frame_index] - palm[:3, 3])
        radial = np.array([-center_palm[0], 0.0, -center_palm[2]])
        radial /= max(np.linalg.norm(radial), 1e-9)
        thumb_radius = 0.021
        finger_radius = 0.027
        current_frame = dict(previous_fingers)
        for finger in range(1, 6):
            joint_names = [f"right_finger{finger}_joint{j}" for j in range(1, 5)]
            tip_name = f"right_finger{finger}_tip_link"
            seed = np.asarray([RIGHT_HAND_GRASP[name] for name in joint_names])
            initial = np.asarray([previous_fingers[name] for name in joint_names])
            finger_lower = np.asarray([joint_lookup[name].lower for name in joint_names])
            finger_upper = np.asarray([joint_lookup[name].upper for name in joint_names])
            target = center_palm + (
                radial * thumb_radius if finger == 1 else -radial * finger_radius
            )
            target[1] = center_palm[1] + finger_y_offsets[finger - 1]

            def finger_residual(values: np.ndarray) -> np.ndarray:
                q = dict(q_body)
                q.update(current_frame)
                q.update(zip(joint_names, values))
                finger_poses = model.forward(transform(x[:3], x[3:6]), q)
                tip = finger_poses[tip_name][:3, 3]
                tip_palm = palm[:3, :3].T @ (tip - palm[:3, 3])
                return np.r_[120.0 * (tip_palm - target), 0.18 * (values - seed), 0.08 * (values - initial)]

            result = least_squares(
                finger_residual,
                initial,
                bounds=(finger_lower, finger_upper),
                max_nfev=1200,
                xtol=1e-11,
                ftol=1e-11,
                gtol=1e-11,
            )
            current_frame.update(zip(joint_names, result.x))
            q_check = dict(q_body)
            q_check.update(current_frame)
            check_poses = model.forward(transform(x[:3], x[3:6]), q_check)
            tip = check_poses[tip_name][:3, 3]
            tip_palm = palm[:3, :3].T @ (tip - palm[:3, 3])
            finger_surface_errors[frame_index, finger - 1] = np.linalg.norm(tip_palm - target)
        for name, value in current_frame.items():
            full_q[frame_index, sim_names.index(name)] = value
        previous_fingers = current_frame

    root_poses = np.zeros((len(xs), 7))
    root_poses[:, :3] = xs[:, :3]
    quat_xyzw = Rotation.from_euler("xyz", xs[:, 3:6]).as_quat()
    root_poses[:, 3:] = quat_xyzw[:, [3, 0, 1, 2]]

    metrics = np.asarray(metrics)
    np.savez(
        OUTPUT,
        door_angle_rad=np.deg2rad(reference["angles_deg"]),
        handle_positions_w=handle_centers,
        handle_extent_w=handle_extent,
        robot_root_pose_wxyz=root_poses,
        robot_joint_names=np.asarray(sim_names),
        robot_joint_positions=full_q,
        palm_position_error_m=metrics[:, 0],
        palm_orientation_error_rad=metrics[:, 1],
        torso_orientation_error_rad=metrics[:, 2],
        com_offset_m=metrics[:, 3],
        fingertip_surface_error_m=finger_surface_errors,
    )

    names = list(sim_names)
    print("angle palm_mm palm_deg torso_deg com_mm waist_yaw knee_l knee_r root_z")
    for i, degrees in enumerate(reference["angles_deg"]):
        value = lambda name: np.rad2deg(full_q[i, names.index(name)])
        print(
            f"{degrees:5.1f} {1000*metrics[i,0]:7.2f} {np.rad2deg(metrics[i,1]):8.2f} "
            f"{np.rad2deg(metrics[i,2]):9.2f} {1000*metrics[i,3]:6.1f} "
            f"{value('waist_yaw_joint'):9.1f} {value('left_knee_joint'):6.1f} "
            f"{value('right_knee_joint'):6.1f} {root_poses[i,2]:6.3f}"
        )
    print(f"HANDLE_CENTER0={handle_center0.tolist()}")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
