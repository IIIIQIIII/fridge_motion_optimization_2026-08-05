#!/usr/bin/env python3
"""Solve fixed-foot keyframes against handle positions exported from Isaac Sim."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from generate_action_keyframes import (
    BODY_JOINTS,
    ROOT,
    URDF_PATH,
    UrdfKinematics,
    pose_error,
    transform,
)


REFERENCE = ROOT / "outputs/scene_reference.json"
OUTPUT = ROOT / "outputs/isaac_aligned_action.npz"


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

    base_margin = np.array([0.28, 0.28, 0.14, 0.28, 0.28, 0.42])
    lower = np.r_[x0[:6] - base_margin, [joint_lookup[name].lower for name in BODY_JOINTS]]
    upper = np.r_[x0[:6] + base_margin, [joint_lookup[name].upper for name in BODY_JOINTS]]
    posture_weights = np.full(len(BODY_JOINTS), 0.55)
    for i, name in enumerate(BODY_JOINTS):
        if name.startswith("right_") and any(part in name for part in ("shoulder", "elbow", "wrist")):
            posture_weights[i] = 0.14
        elif name.startswith("left_") and any(part in name for part in ("shoulder", "elbow", "wrist")):
            posture_weights[i] = 1.0
        elif name.startswith("waist_"):
            posture_weights[i] = 0.22

    targets = np.asarray(reference["handle_positions_w"], dtype=float)
    angles = np.deg2rad(reference["angles_deg"])
    solutions = []
    palm_errors = []
    foot_errors = []
    previous = x0.copy()

    for frame_idx, target in enumerate(targets):
        def residual(x):
            poses = model.forward(*unpack(x))
            palm = poses["right_palm_link"][:3, 3]
            com = model.center_of_mass(poses)
            parts = [55.0 * (palm - target)]
            for foot_name in foot_names:
                err = pose_error(poses[foot_name], foot_targets[foot_name])
                parts.append(np.r_[80.0 * err[:3], 32.0 * err[3:]])
            parts.append(2.2 * (com[:2] - support_center))
            parts.append(np.array([1.0, 1.0, 1.4, 0.9, 0.9, 0.8]) * (x[:6] - x0[:6]))
            parts.append(posture_weights * (x[6:] - x0[6:]))
            if frame_idx:
                parts.append(np.r_[np.ones(6) * 1.0, np.ones(len(BODY_JOINTS)) * 0.25] * (x - previous))
            return np.concatenate(parts)

        result = least_squares(
            residual,
            previous,
            bounds=(lower, upper),
            max_nfev=1800,
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
        )
        poses = model.forward(*unpack(result.x))
        solutions.append(result.x.copy())
        palm_errors.append(np.linalg.norm(poses["right_palm_link"][:3, 3] - target))
        foot_errors.append(max(np.linalg.norm(poses[name][:3, 3] - foot_targets[name][:3, 3]) for name in foot_names))
        previous = result.x.copy()

    xs = np.vstack(solutions)
    full_joint_q = np.repeat(sim_q0[None, :], len(xs), axis=0)
    for body_index, name in enumerate(BODY_JOINTS):
        full_joint_q[:, sim_names.index(name)] = xs[:, 6 + body_index]

    root_poses = np.zeros((len(xs), 7))
    root_poses[:, :3] = xs[:, :3]
    xyzw = Rotation.from_euler("xyz", xs[:, 3:6]).as_quat()
    root_poses[:, 3:] = xyzw[:, [3, 0, 1, 2]]

    np.savez(
        OUTPUT,
        door_angle_rad=angles,
        handle_positions_w=targets,
        robot_root_pose_wxyz=root_poses,
        robot_joint_names=np.asarray(sim_names),
        robot_joint_positions=full_joint_q,
        palm_error_m=np.asarray(palm_errors),
        foot_error_m=np.asarray(foot_errors),
    )
    print("angle_deg palm_error_mm foot_error_mm")
    for angle, palm_error, foot_error in zip(np.rad2deg(angles), palm_errors, foot_errors):
        print(f"{angle:8.1f} {1000*palm_error:13.3f} {1000*foot_error:13.3f}")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
