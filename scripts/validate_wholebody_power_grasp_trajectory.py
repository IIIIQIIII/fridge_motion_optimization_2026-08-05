#!/usr/bin/env python3
"""Strict dense-frame validation for the full-body power-grasp trajectory."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation, Slerp

from generate_action_keyframes import BODY_JOINTS, ROOT, URDF_PATH, UrdfKinematics, pose_error
from optimize_static_power_grasp_geometry import GraspModel, load_stl_surface


TRAJECTORY = ROOT / "outputs/isaac_wholebody_power_grasp_action.npz"
REFERENCE = ROOT / "outputs/scene_reference.json"
GRASP_SOLUTION = ROOT / "outputs/static_power_grasp_geometry_solution.json"
OUTPUT = ROOT / "outputs/wholebody_power_grasp_strict_validation.json"
DENSE_FRAMES = 241
MOTION_DURATION_S = 10.0


def minimum_jerk(value: float) -> float:
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def pose_from_wxyz(values: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = values[:3]
    pose[:3, :3] = Rotation.from_quat(values[[4, 5, 6, 3]]).as_matrix()
    return pose


def interpolate_trajectory(data: np.lib.npyio.NpzFile):
    key_count = len(data["door_angle_rad"])
    positions = np.linspace(0.0, key_count - 1, DENSE_FRAMES)
    keys = np.arange(key_count, dtype=float)
    roots = np.zeros((DENSE_FRAMES, 7))
    roots[:, :3] = CubicSpline(keys, data["robot_root_pose_wxyz"][:, :3], bc_type="natural")(positions)
    roots[:, 3:] = Slerp(
        keys, Rotation.from_quat(data["robot_root_pose_wxyz"][:, [4, 5, 6, 3]])
    )(positions).as_quat()[:, [3, 0, 1, 2]]
    joints = CubicSpline(keys, data["robot_joint_positions"], bc_type="natural")(positions)
    doors = CubicSpline(keys, data["door_angle_rad"], bc_type="natural")(positions)
    rolls = CubicSpline(keys, data["grasp_axial_roll_rad"], bc_type="natural")(positions)
    foot_targets = np.zeros((DENSE_FRAMES, len(data["foot_target_names"]), 7))
    for foot_index in range(len(data["foot_target_names"])):
        values = data["foot_target_poses_wxyz"][:, foot_index]
        foot_targets[:, foot_index, :3] = CubicSpline(
            keys, values[:, :3], bc_type="natural"
        )(positions)
        foot_targets[:, foot_index, 3:] = Slerp(
            keys, Rotation.from_quat(values[:, [4, 5, 6, 3]])
        )(positions).as_quat()[:, [3, 0, 1, 2]]
    return roots, joints, doors, rolls, foot_targets


def hand_parameter_vector(q: dict[str, float], center_in_palm: np.ndarray) -> np.ndarray:
    return np.r_[
        center_in_palm,
        [q[f"right_finger1_joint{i}"] for i in range(1, 5)],
        [q[f"right_finger{finger}_joint{joint}"] for finger in range(2, 6) for joint in (1, 3, 4)],
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, default=TRAJECTORY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    trajectory = np.load(args.trajectory)
    reference = json.loads(REFERENCE.read_text())
    grasp = json.loads(GRASP_SOLUTION.read_text())
    names = [str(name) for name in trajectory["robot_joint_names"]]
    name_index = {name: index for index, name in enumerate(names)}
    model = UrdfKinematics(URDF_PATH)
    joint_lookup = {joint.name: joint for joint in model.joints}
    roots, joints, doors, rolls, moving_foot_targets = interpolate_trajectory(trajectory)

    reference_angles = np.asarray(reference["angles_deg"], dtype=float)
    reference_doors = np.asarray(reference["door_body_poses_wxyz"], dtype=float)
    door_slerp = Slerp(
        reference_angles, Rotation.from_quat(reference_doors[:, [4, 5, 6, 3]])
    )
    door0 = pose_from_wxyz(reference_doors[0])
    handle_center0 = np.asarray(grasp["measured_handle_axis_center_w"], dtype=float)
    handle0_h = np.r_[handle_center0, 1.0]

    q0 = dict(zip(names, joints[0]))
    poses0 = model.forward(pose_from_wxyz(roots[0]), q0)
    palm0 = poses0["right_palm_link"].copy()
    handle_in_palm = poses0["right_palm_link"][:3, :3].T @ (
        handle_center0 - poses0["right_palm_link"][:3, 3]
    )
    foot_names = ("left_ankle_roll_link", "right_ankle_roll_link")
    foot_target_names = [str(name) for name in trajectory["foot_target_names"]]
    if foot_target_names != list(foot_names):
        raise RuntimeError(f"Unexpected foot target order: {foot_target_names}")
    torso0 = poses0["torso_link"].copy()

    # Foot surface samples are transformed on every frame because this is a
    # mobile whole-body trajectory rather than a fixed-support special case.
    foot_surfaces = {}
    for name in foot_names:
        mesh_path, visual_tf, _ = model.visuals[name]
        foot_surfaces[name] = (load_stl_surface(mesh_path, None), visual_tf)

    hand_model = GraspModel(sampled_points=None)
    records = []
    for frame_index, (root, q_values, door_angle, roll, foot_target_values) in enumerate(
        zip(roots, joints, doors, rolls, moving_foot_targets)
    ):
        angle_deg = float(np.rad2deg(door_angle))
        clipped_angle = float(np.clip(angle_deg, reference_angles[0], reference_angles[-1]))
        door = np.eye(4)
        door[:3, 3] = [
            np.interp(clipped_angle, reference_angles, reference_doors[:, axis]) for axis in range(3)
        ]
        door[:3, :3] = door_slerp([clipped_angle]).as_matrix()[0]
        door_delta = door @ np.linalg.inv(door0)
        handle_center = (door_delta @ handle0_h)[:3]
        target_R = Rotation.from_rotvec([0.0, 0.0, roll]).as_matrix() @ door_delta[:3, :3] @ palm0[:3, :3]
        target_p = handle_center - target_R @ handle_in_palm

        q = dict(zip(names, q_values))
        poses = model.forward(pose_from_wxyz(root), q)
        palm = poses["right_palm_link"]
        palm_position_error = float(np.linalg.norm(palm[:3, 3] - target_p))
        palm_orientation_error = float(
            Rotation.from_matrix(target_R.T @ palm[:3, :3]).magnitude()
        )
        frame_foot_targets = {
            name: pose_from_wxyz(values)
            for name, values in zip(foot_names, foot_target_values)
        }
        foot_errors = [pose_error(poses[name], frame_foot_targets[name]) for name in foot_names]
        torso_error = float(
            Rotation.from_matrix(torso0[:3, :3].T @ poses["torso_link"][:3, :3]).magnitude()
        )
        foot_xy = []
        for name in foot_names:
            points, visual_tf = foot_surfaces[name]
            tf = poses[name] @ visual_tf
            world = points @ tf[:3, :3].T + tf[:3, 3]
            foot_xy.append(world[:, :2])
        support_hull = ConvexHull(np.concatenate(foot_xy))
        hull_a = support_hull.equations[:, :2]
        hull_b = support_hull.equations[:, 2]
        com = model.center_of_mass(poses)
        support_margin = float(np.min(-(hull_a @ com[:2] + hull_b) / np.linalg.norm(hull_a, axis=1)))

        center_in_palm = palm[:3, :3].T @ (handle_center - palm[:3, 3])
        hand_model.axis = palm[:3, :3].T @ np.array([0.0, 0.0, 1.0])
        hand_model.axis /= np.linalg.norm(hand_model.axis)
        hand = hand_model.metrics(hand_parameter_vector(q, center_in_palm))
        contact_gap = max(hand["link_metrics"][name]["patch_gap_m"] for name in hand["required_contact_links"])
        near_gap = max(hand["link_metrics"][name]["patch_gap_m"] for name in hand["required_near_links"])
        collision_metrics = [hand["group_metrics"]["palm"], *hand["link_metrics"].values()]
        penetration = max(metric["max_penetration_m"] for metric in collision_metrics)
        shoulder = poses["right_shoulder_roll_link"][:3, 3]
        elbow = poses["right_elbow_link"][:3, 3]
        wrist = poses["right_wrist_yaw_link"][:3, 3]
        shoulder_to_elbow = shoulder - elbow
        wrist_to_elbow = wrist - elbow
        elbow_bend = float(
            np.rad2deg(
                np.arccos(
                    np.clip(
                        np.dot(shoulder_to_elbow, wrist_to_elbow)
                        / (np.linalg.norm(shoulder_to_elbow) * np.linalg.norm(wrist_to_elbow)),
                        -1.0,
                        1.0,
                    )
                )
            )
        )
        torso_yaw = float(np.rad2deg(Rotation.from_matrix(poses["torso_link"][:3, :3]).as_euler("xyz")[2]))

        records.append(
            {
                "frame": frame_index,
                "door_angle_deg": angle_deg,
                "palm_position_error_m": palm_position_error,
                "palm_orientation_error_deg": float(np.rad2deg(palm_orientation_error)),
                "foot_position_error_m": max(float(np.linalg.norm(error[:3])) for error in foot_errors),
                "foot_orientation_error_deg": max(float(np.rad2deg(np.linalg.norm(error[3:]))) for error in foot_errors),
                "torso_orientation_error_deg": float(np.rad2deg(torso_error)),
                "support_margin_m": support_margin,
                "palm_surface_gap_m": hand["group_metrics"]["palm"]["patch_gap_m"],
                "contact_link_gap_m": float(contact_gap),
                "near_link_gap_m": float(near_gap),
                "max_hand_penetration_m": float(penetration),
                "enclosure_angle_deg": hand["enclosure_angle_deg"],
                "waist_yaw_deg": float(np.rad2deg(q["waist_yaw_joint"])),
                "waist_roll_deg": float(np.rad2deg(q["waist_roll_joint"])),
                "waist_pitch_deg": float(np.rad2deg(q["waist_pitch_joint"])),
                "left_knee_deg": float(np.rad2deg(q["left_knee_joint"])),
                "right_knee_deg": float(np.rad2deg(q["right_knee_joint"])),
                "right_elbow_deg": float(np.rad2deg(q["right_elbow_joint"])),
                "right_wrist_roll_deg": float(np.rad2deg(q["right_wrist_roll_joint"])),
                "right_wrist_pitch_deg": float(np.rad2deg(q["right_wrist_pitch_joint"])),
                "right_wrist_yaw_deg": float(np.rad2deg(q["right_wrist_yaw_joint"])),
                "right_shoulder_pitch_deg": float(np.rad2deg(q["right_shoulder_pitch_joint"])),
                "right_shoulder_roll_deg": float(np.rad2deg(q["right_shoulder_roll_joint"])),
                "right_shoulder_yaw_deg": float(np.rad2deg(q["right_shoulder_yaw_joint"])),
                "elbow_bend_angle_deg": elbow_bend,
                "shoulder_to_palm_m": float(np.linalg.norm(shoulder - palm[:3, 3])),
                "torso_yaw_deg": torso_yaw,
            }
        )

    dt = MOTION_DURATION_S / (DENSE_FRAMES - 1)
    body_indices = [name_index[name] for name in BODY_JOINTS]
    body_velocity = np.diff(joints[:, body_indices], axis=0) / dt
    body_acceleration = np.diff(body_velocity, axis=0) / dt
    root_speed = np.linalg.norm(np.diff(roots[:, :3], axis=0), axis=1) / dt
    key_joint_delta = np.max(np.abs(np.diff(trajectory["robot_joint_positions"][:, body_indices], axis=0)))
    key_roll_delta = np.max(np.abs(np.diff(trajectory["grasp_axial_roll_rad"])))
    key_root_delta = np.max(
        np.linalg.norm(np.diff(trajectory["robot_root_pose_wxyz"][:, :3], axis=0), axis=1)
    )
    foot_path_delta = np.linalg.norm(
        moving_foot_targets[-1, :, :3] - moving_foot_targets[0, :, :3], axis=1
    )

    joint_limit_violation = 0.0
    for name in BODY_JOINTS:
        values = joints[:, name_index[name]]
        joint = joint_lookup[name]
        joint_limit_violation = max(
            joint_limit_violation,
            float(max(0.0, joint.lower - values.min(), values.max() - joint.upper)),
        )

    def maximum(field: str) -> float:
        return max(float(record[field]) for record in records)

    def minimum(field: str) -> float:
        return min(float(record[field]) for record in records)

    thresholds = {
        "palm_position_error_m": 0.0010,
        "palm_orientation_error_deg": 1.0,
        "foot_position_error_m": 0.0020,
        "foot_orientation_error_deg": 0.30,
        "torso_orientation_error_deg": 20.0,
        "support_margin_m": 0.010,
        "palm_surface_gap_m": 0.0060,
        "contact_link_gap_m": 0.0020,
        "near_link_gap_m": 0.0070,
        "max_hand_penetration_m": 0.0010,
        "enclosure_angle_deg": 190.0,
        "absolute_waist_yaw_deg": 8.0,
        "absolute_waist_roll_deg": 10.0,
        "absolute_waist_pitch_deg": 10.0,
        "minimum_knee_deg": 5.0,
        "maximum_knee_deg": 30.0,
        "absolute_wrist_roll_deg": 45.0,
        "absolute_wrist_pitch_deg": 45.0,
        "absolute_wrist_yaw_deg": 130.0,
        "absolute_shoulder_pitch_deg": 90.0,
        "absolute_shoulder_roll_deg": 60.0,
        "absolute_shoulder_yaw_deg": 70.0,
        "minimum_elbow_bend_angle_deg": 70.0,
        "maximum_elbow_bend_angle_deg": 175.0,
        "minimum_shoulder_to_palm_m": 0.20,
        "maximum_shoulder_to_palm_m": 0.42,
        "minimum_final_torso_turn_deg": 0.18 * float(np.rad2deg(trajectory["door_angle_rad"][-1])),
        "minimum_elbow_deg": -10.1,
        "joint_limit_violation_rad": 1e-8,
        "max_body_velocity_rad_s": 4.0,
        "max_body_acceleration_rad_s2": 35.0,
        "max_root_speed_m_s": 0.8,
        "max_key_joint_delta_rad": np.deg2rad(7.0),
        "max_key_root_delta_m": 0.012,
        "max_key_grasp_roll_delta_rad": np.deg2rad(5.0),
        "max_foot_path_displacement_m": 0.25,
    }
    observed = {
        "palm_position_error_m": maximum("palm_position_error_m"),
        "palm_orientation_error_deg": maximum("palm_orientation_error_deg"),
        "foot_position_error_m": maximum("foot_position_error_m"),
        "foot_orientation_error_deg": maximum("foot_orientation_error_deg"),
        "torso_orientation_error_deg": maximum("torso_orientation_error_deg"),
        "support_margin_m": minimum("support_margin_m"),
        "palm_surface_gap_m": maximum("palm_surface_gap_m"),
        "contact_link_gap_m": maximum("contact_link_gap_m"),
        "near_link_gap_m": maximum("near_link_gap_m"),
        "max_hand_penetration_m": maximum("max_hand_penetration_m"),
        "enclosure_angle_deg": minimum("enclosure_angle_deg"),
        "absolute_waist_yaw_deg": max(abs(record["waist_yaw_deg"]) for record in records),
        "absolute_waist_roll_deg": max(abs(record["waist_roll_deg"]) for record in records),
        "absolute_waist_pitch_deg": max(abs(record["waist_pitch_deg"]) for record in records),
        "minimum_knee_deg": min(min(record["left_knee_deg"], record["right_knee_deg"]) for record in records),
        "maximum_knee_deg": max(max(record["left_knee_deg"], record["right_knee_deg"]) for record in records),
        "absolute_wrist_roll_deg": max(abs(record["right_wrist_roll_deg"]) for record in records),
        "absolute_wrist_pitch_deg": max(abs(record["right_wrist_pitch_deg"]) for record in records),
        "absolute_wrist_yaw_deg": max(abs(record["right_wrist_yaw_deg"]) for record in records),
        "absolute_shoulder_pitch_deg": max(abs(record["right_shoulder_pitch_deg"]) for record in records),
        "absolute_shoulder_roll_deg": max(abs(record["right_shoulder_roll_deg"]) for record in records),
        "absolute_shoulder_yaw_deg": max(abs(record["right_shoulder_yaw_deg"]) for record in records),
        "minimum_elbow_bend_angle_deg": minimum("elbow_bend_angle_deg"),
        "maximum_elbow_bend_angle_deg": maximum("elbow_bend_angle_deg"),
        "minimum_shoulder_to_palm_m": minimum("shoulder_to_palm_m"),
        "maximum_shoulder_to_palm_m": maximum("shoulder_to_palm_m"),
        "minimum_final_torso_turn_deg": abs(records[-1]["torso_yaw_deg"] - records[0]["torso_yaw_deg"]),
        "minimum_elbow_deg": min(record["right_elbow_deg"] for record in records),
        "joint_limit_violation_rad": joint_limit_violation,
        "max_body_velocity_rad_s": float(np.max(np.abs(body_velocity))),
        "max_body_acceleration_rad_s2": float(np.max(np.abs(body_acceleration))),
        "max_root_speed_m_s": float(np.max(root_speed)),
        "max_key_joint_delta_rad": float(key_joint_delta),
        "max_key_root_delta_m": float(key_root_delta),
        "max_key_grasp_roll_delta_rad": float(key_roll_delta),
        "max_foot_path_displacement_m": float(np.max(foot_path_delta)),
    }
    lower_bound_metrics = {
        "support_margin_m",
        "enclosure_angle_deg",
        "minimum_knee_deg",
        "minimum_elbow_deg",
        "minimum_elbow_bend_angle_deg",
        "minimum_shoulder_to_palm_m",
        "minimum_final_torso_turn_deg",
    }
    tests = {}
    for name, threshold in thresholds.items():
        passed = observed[name] >= threshold if name in lower_bound_metrics else observed[name] <= threshold
        tests[name] = {"observed": observed[name], "threshold": threshold, "passed": bool(passed)}
    accepted = all(test["passed"] for test in tests.values())
    report = {
        "accepted": accepted,
        "dense_frames": DENSE_FRAMES,
        "motion_duration_s": MOTION_DURATION_S,
        "tests": tests,
        "frames": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"accepted={accepted} dense_frames={DENSE_FRAMES}")
    for name, test in tests.items():
        print(f"{'PASS' if test['passed'] else 'FAIL'} {name}: observed={test['observed']:.6g} threshold={test['threshold']:.6g}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
