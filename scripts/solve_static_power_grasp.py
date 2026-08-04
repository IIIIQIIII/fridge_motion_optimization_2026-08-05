#!/usr/bin/env python3
"""Fit one palm-seated power grasp for the upper refrigerator handle."""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from generate_action_keyframes import BODY_JOINTS, ROOT, URDF_PATH, UrdfKinematics, pose_error, transform


SOURCE = ROOT / "outputs/isaac_humanlike_grasp_refined2.npz"
REFERENCE = ROOT / "outputs/scene_reference.json"
GEOMETRY = ROOT / "outputs/fridge_handle_geometry.json"
GRASP_SOLUTION = ROOT / "outputs/static_power_grasp_geometry_solution.json"
OUTPUT = ROOT / "outputs/isaac_static_power_grasp.npz"

def pose_from_wxyz(values: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, 3] = values[:3]
    w, x, y, z = values[3:]
    out[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    return out


def main() -> None:
    source = np.load(SOURCE)
    reference = json.loads(REFERENCE.read_text())
    geometry = json.loads(GEOMETRY.read_text())
    grasp_solution = json.loads(GRASP_SOLUTION.read_text())
    if not grasp_solution.get("accepted", False):
        raise RuntimeError("Static grasp geometry has not passed its acceptance tests")
    # Use the vertical bar axis, measured from the true USD mesh cross-section.
    # The parent Xform center is shifted by the two mounting brackets and is not
    # the center of the surface that the hand grasps.
    handle_center = np.asarray(grasp_solution["measured_handle_axis_center_w"], dtype=float)
    right_power_grasp = {
        name: float(value) for name, value in grasp_solution["joint_positions_rad"].items()
    }
    model = UrdfKinematics(URDF_PATH)
    joint_lookup = {joint.name: joint for joint in model.joints}
    sim_names = [str(name) for name in source["robot_joint_names"]]
    full_q0 = source["robot_joint_positions"][0].copy()
    q0 = dict(zip(sim_names, full_q0))
    root0 = source["robot_root_pose_wxyz"][0].copy()
    root_euler = Rotation.from_quat([root0[4], root0[5], root0[6], root0[3]]).as_euler("xyz")
    x0 = np.r_[root0[:3], root_euler, [q0[name] for name in BODY_JOINTS]]

    def unpack(x: np.ndarray):
        return transform(x[:3], x[3:6]), dict(zip(BODY_JOINTS, x[6:]))

    initial_poses = model.forward(*unpack(x0))
    palm_R_target = initial_poses["right_palm_link"][:3, :3].copy()
    handle_in_palm = np.asarray(grasp_solution["center_in_palm_m"], dtype=float)
    palm_position_target = handle_center - palm_R_target @ handle_in_palm
    foot_names = ("left_ankle_roll_link", "right_ankle_roll_link")
    foot_targets = {name: initial_poses[name].copy() for name in foot_names}
    torso_target = initial_poses["torso_link"].copy()
    support_center = np.mean([foot_targets[name][:2, 3] for name in foot_names], axis=0)

    lower = np.r_[
        x0[:3] - np.array([0.05, 0.05, 0.025]),
        x0[3:6] - np.deg2rad([4.0, 4.0, 6.0]),
        [joint_lookup[name].lower for name in BODY_JOINTS],
    ]
    upper = np.r_[
        x0[:3] + np.array([0.05, 0.05, 0.025]),
        x0[3:6] + np.deg2rad([4.0, 4.0, 6.0]),
        [joint_lookup[name].upper for name in BODY_JOINTS],
    ]
    index = {name: 6 + i for i, name in enumerate(BODY_JOINTS)}
    for name, margin in (
        ("waist_yaw_joint", 6.0),
        ("waist_roll_joint", 6.0),
        ("waist_pitch_joint", 6.0),
        ("right_wrist_roll_joint", 8.0),
        ("right_wrist_pitch_joint", 8.0),
        ("right_wrist_yaw_joint", 20.0),
    ):
        i = index[name]
        lower[i] = max(lower[i], x0[i] - np.deg2rad(margin))
        upper[i] = min(upper[i], x0[i] + np.deg2rad(margin))
    for name in BODY_JOINTS:
        if any(part in name for part in ("hip", "knee", "ankle")):
            i = index[name]
            lower[i] = max(lower[i], x0[i] - np.deg2rad(4.0))
            upper[i] = min(upper[i], x0[i] + np.deg2rad(4.0))

    posture_weights = np.full(len(BODY_JOINTS), 0.5)
    for i, name in enumerate(BODY_JOINTS):
        if any(part in name for part in ("hip", "knee", "ankle")):
            posture_weights[i] = 5.0
        elif name.startswith("waist_"):
            posture_weights[i] = 5.0
        elif name.startswith("left_"):
            posture_weights[i] = 3.0
        elif "wrist" in name:
            posture_weights[i] = 2.0
        elif name.startswith("right_"):
            posture_weights[i] = 0.12

    def residual(x: np.ndarray) -> np.ndarray:
        poses = model.forward(*unpack(x))
        palm = poses["right_palm_link"]
        palm_pos = palm[:3, 3] - palm_position_target
        palm_rot = Rotation.from_matrix(palm_R_target.T @ palm[:3, :3]).as_rotvec()
        torso = pose_error(poses["torso_link"], torso_target)
        # Millimetres matter at the grasp surface: make the palm target much
        # stiffer than posture regularization while keeping both feet fixed.
        parts = [450.0 * palm_pos, 28.0 * palm_rot]
        for name in foot_names:
            err = pose_error(poses[name], foot_targets[name])
            parts.append(np.r_[120.0 * err[:3], 40.0 * err[3:]])
        parts.extend(
            [
                np.r_[8.0 * torso[:3], 12.0 * torso[3:]],
                15.0 * (model.center_of_mass(poses)[:2] - support_center),
                posture_weights * (x[6:] - x0[6:]),
                np.array([2.0, 2.0, 2.5, 4.0, 4.0, 3.0]) * (x[:6] - x0[:6]),
            ]
        )
        return np.concatenate(parts)

    result = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        max_nfev=3000,
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
    )
    poses = model.forward(*unpack(result.x))
    palm = poses["right_palm_link"]
    actual_handle_in_palm = palm[:3, :3].T @ (handle_center - palm[:3, 3])

    full_q = full_q0.copy()
    for i, name in enumerate(BODY_JOINTS):
        full_q[sim_names.index(name)] = result.x[6 + i]
    for name, value in right_power_grasp.items():
        full_q[sim_names.index(name)] = value
    root_quat = Rotation.from_euler("xyz", result.x[3:6]).as_quat()
    root = np.r_[result.x[:3], root_quat[[3, 0, 1, 2]]]

    np.savez(
        OUTPUT,
        door_angle_rad=np.array([0.0, 0.0]),
        handle_positions_w=np.repeat(handle_center[None, :], 2, axis=0),
        robot_root_pose_wxyz=np.repeat(root[None, :], 2, axis=0),
        robot_joint_names=np.asarray(sim_names),
        robot_joint_positions=np.repeat(full_q[None, :], 2, axis=0),
        handle_in_palm_target=handle_in_palm,
        handle_in_palm_actual=actual_handle_in_palm,
    )
    print(f"handle_in_palm_target={np.round(handle_in_palm, 4).tolist()}")
    print(f"handle_in_palm_actual={np.round(actual_handle_in_palm, 4).tolist()}")
    print(f"palm_position_error_mm={1000*np.linalg.norm(palm[:3,3]-palm_position_target):.2f}")
    print(f"palm_orientation_error_deg={np.rad2deg(np.linalg.norm(Rotation.from_matrix(palm_R_target.T@palm[:3,:3]).as_rotvec())):.2f}")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
