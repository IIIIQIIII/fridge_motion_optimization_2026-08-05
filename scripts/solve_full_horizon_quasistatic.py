#!/usr/bin/env python3
"""Jointly optimize the complete fixed-support refrigerator-opening horizon.

Unlike the legacy greedy per-angle IK, every node is one optimization variable.
Both feet remain fixed in the world frame. If the closed chain becomes
infeasible, the solver must report it; it may not translate both feet/root to
hide the need for a contact-mode change.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from math_model.horizon import joint_limit_barrier
from generate_action_keyframes import BODY_JOINTS, ROOT as PROJECT_ROOT, URDF_PATH, UrdfKinematics, pose_error, transform

DEFAULT_INPUT = PROJECT_ROOT / "outputs/isaac_wholebody_power_grasp_action.npz"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/isaac_full_horizon_fixed_support_action.npz"
FOOT_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")


def pose_from_wxyz(v: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = v[:3]
    pose[:3, :3] = Rotation.from_quat(v[[4, 5, 6, 3]]).as_matrix()
    return pose


def pose_to_wxyz(pose: np.ndarray) -> np.ndarray:
    return np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]]


def state_from_frame(root: np.ndarray, names: list[str], q: np.ndarray) -> np.ndarray:
    euler = Rotation.from_quat(root[[4, 5, 6, 3]]).as_euler("xyz")
    lookup = dict(zip(names, q))
    return np.r_[root[:3], euler, [lookup[name] for name in BODY_JOINTS]]


def soft_interval(value: float, lower: float, upper: float, weight: float) -> np.ndarray:
    sharpness = 25.0
    low = np.logaddexp(0.0, sharpness * (lower - value)) / sharpness
    high = np.logaddexp(0.0, sharpness * (value - upper)) / sharpness
    return weight * np.array([low, high])


def sparsity(frame_count: int, state_size: int, local_size: int) -> lil_matrix:
    rows = frame_count * local_size
    rows += max(frame_count - 1, 0) * state_size
    rows += max(frame_count - 2, 0) * state_size
    rows += max(frame_count - 3, 0) * state_size
    out = lil_matrix((rows, frame_count * state_size), dtype=int)
    row = 0
    for frame in range(frame_count):
        out[row:row + local_size, frame * state_size:(frame + 1) * state_size] = 1
        row += local_size
    for width, count in ((2, frame_count - 1), (3, frame_count - 2), (4, frame_count - 3)):
        for start in range(max(count, 0)):
            for frame in range(start, start + width):
                out[row:row + state_size, frame * state_size:(frame + 1) * state_size] = 1
            row += state_size
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nodes", type=int, default=13)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--max-nfev", type=int, default=250)
    args = parser.parse_args()
    if args.nodes < 5 or args.duration <= 0.0:
        raise ValueError("nodes >= 5 and duration > 0 are required")

    source = np.load(args.input)
    required = {"door_angle_rad", "robot_root_pose_wxyz", "robot_joint_names", "robot_joint_positions", "palm_target_poses_wxyz"}
    missing = sorted(required.difference(source.files))
    if missing:
        raise KeyError(f"input trajectory is missing: {missing}")

    names = [str(name) for name in source["robot_joint_names"]]
    indices = np.unique(np.round(np.linspace(0, len(source["door_angle_rad"]) - 1, args.nodes)).astype(int))
    frame_count = len(indices)
    angles = source["door_angle_rad"][indices]
    roots_seed = source["robot_root_pose_wxyz"][indices]
    joints_seed = source["robot_joint_positions"][indices]
    palm_targets = np.asarray([pose_from_wxyz(v) for v in source["palm_target_poses_wxyz"][indices]])

    model = UrdfKinematics(URDF_PATH)
    joint_lookup = {joint.name: joint for joint in model.joints}
    state_size = 6 + len(BODY_JOINTS)
    seed = np.vstack([state_from_frame(root, names, q) for root, q in zip(roots_seed, joints_seed)])

    def unpack(state: np.ndarray):
        return transform(state[:3], state[3:6]), dict(zip(BODY_JOINTS, state[6:]))

    initial_poses = model.forward(*unpack(seed[0]))
    foot_targets = {name: initial_poses[name].copy() for name in FOOT_NAMES}
    support_center = np.mean([pose[:2, 3] for pose in foot_targets.values()], axis=0)
    torso_initial = initial_poses["torso_link"].copy()
    root_initial, posture_initial = seed[0, :6].copy(), seed[0, 6:].copy()
    lower_joint = np.asarray([joint_lookup[name].lower for name in BODY_JOINTS])
    upper_joint = np.asarray([joint_lookup[name].upper for name in BODY_JOINTS])
    root_delta = np.r_[0.20, 0.16, 0.06, np.deg2rad([10.0, 15.0, 35.0])]
    lower_state = np.r_[root_initial - root_delta, lower_joint]
    upper_state = np.r_[root_initial + root_delta, upper_joint]
    lower, upper = np.tile(lower_state, frame_count), np.tile(upper_state, frame_count)

    posture_weight = np.full(len(BODY_JOINTS), 0.35)
    for i, name in enumerate(BODY_JOINTS):
        if name.startswith("left_"):
            posture_weight[i] = 1.8
        if any(token in name for token in ("hip", "knee", "ankle")):
            posture_weight[i] = 1.2
        if name.startswith("waist_"):
            posture_weight[i] = 0.18
        if name.startswith("right_") and any(token in name for token in ("shoulder", "elbow", "wrist")):
            posture_weight[i] = 0.08

    local_size = 6 + 12 + 2 + 2 + 6 + len(BODY_JOINTS) + len(BODY_JOINTS) + 2 + 2
    dt = args.duration / (frame_count - 1)

    def local_residual(state: np.ndarray, frame: int) -> np.ndarray:
        poses = model.forward(*unpack(state))
        palm, target = poses["right_palm_link"], palm_targets[frame]
        parts = [
            950.0 * (palm[:3, 3] - target[:3, 3]),
            75.0 * Rotation.from_matrix(target[:3, :3].T @ palm[:3, :3]).as_rotvec(),
        ]
        for name in FOOT_NAMES:
            error = pose_error(poses[name], foot_targets[name])
            parts.append(np.r_[900.0 * error[:3], 180.0 * error[3:]])
        parts.append(35.0 * (model.center_of_mass(poses)[:2] - support_center))
        torso_delta = Rotation.from_matrix(torso_initial[:3, :3].T @ poses["torso_link"][:3, :3]).as_rotvec()
        parts.append(15.0 * torso_delta[:2])
        parts.append(np.array([0.5, 0.5, 2.5, 4.0, 4.0, 0.35]) * (state[:6] - root_initial))
        parts.append(posture_weight * (state[6:] - posture_initial))
        parts.append(18.0 * joint_limit_barrier(state[6:], lower_joint, upper_joint))

        shoulder = poses["right_shoulder_roll_link"][:3, 3]
        elbow = poses["right_elbow_link"][:3, 3]
        wrist = poses["right_wrist_yaw_link"][:3, 3]
        a, b = shoulder - elbow, wrist - elbow
        angle = float(np.arccos(np.clip(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9), -1.0, 1.0)))
        reach = float(np.linalg.norm(shoulder - palm[:3, 3]))
        parts.append(soft_interval(angle, np.deg2rad(78.0), np.deg2rad(165.0), 3.0))
        parts.append(soft_interval(reach, 0.23, 0.385, 22.0))
        return np.concatenate(parts)

    def residual(flat: np.ndarray) -> np.ndarray:
        states = flat.reshape(frame_count, state_size)
        parts = [local_residual(states[k], k) for k in range(frame_count)]
        scale = np.r_[np.full(3, 4.0), np.full(3, 1.0), np.ones(len(BODY_JOINTS))]
        velocity = np.diff(states, axis=0) / dt
        acceleration = np.diff(velocity, axis=0) / dt
        jerk = np.diff(acceleration, axis=0) / dt
        parts.extend([0.035 * (velocity * scale).ravel(), 0.055 * (acceleration * scale).ravel(), 0.018 * (jerk * scale).ravel()])
        return np.concatenate(parts)

    result = least_squares(
        residual,
        np.clip(seed.ravel(), lower + 1e-9, upper - 1e-9),
        bounds=(lower, upper),
        jac_sparsity=sparsity(frame_count, state_size, local_size),
        tr_solver="lsmr",
        x_scale="jac",
        max_nfev=args.max_nfev,
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
        verbose=1,
    )
    states = result.x.reshape(frame_count, state_size)
    full_q = np.repeat(joints_seed[0][None, :], frame_count, axis=0)
    roots = np.zeros((frame_count, 7))
    metrics, foot_poses = [], []
    for frame, state in enumerate(states):
        for j, name in enumerate(BODY_JOINTS):
            full_q[frame, names.index(name)] = state[6 + j]
        for j, name in enumerate(names):
            if "finger" in name:
                full_q[frame, j] = joints_seed[frame, j]
        roots[frame, :3] = state[:3]
        roots[frame, 3:] = Rotation.from_euler("xyz", state[3:6]).as_quat()[[3, 0, 1, 2]]
        poses = model.forward(*unpack(state))
        palm, target = poses["right_palm_link"], palm_targets[frame]
        feet_error = [pose_error(poses[name], foot_targets[name]) for name in FOOT_NAMES]
        metrics.append([
            np.linalg.norm(palm[:3, 3] - target[:3, 3]),
            Rotation.from_matrix(target[:3, :3].T @ palm[:3, :3]).magnitude(),
            max(np.linalg.norm(error[:3]) for error in feet_error),
            max(np.linalg.norm(error[3:]) for error in feet_error),
            np.linalg.norm(model.center_of_mass(poses)[:2] - support_center),
        ])
        foot_poses.append([pose_to_wxyz(poses[name]) for name in FOOT_NAMES])
    metrics = np.asarray(metrics)

    frame_arrays = {key for key in source.files if np.ndim(source[key]) > 0 and len(source[key]) == len(source["door_angle_rad"])}
    payload = {key: (source[key][indices] if key in frame_arrays else source[key]) for key in source.files}
    payload.update(
        door_angle_rad=angles,
        robot_root_pose_wxyz=roots,
        robot_joint_positions=full_q,
        palm_target_poses_wxyz=np.asarray([pose_to_wxyz(pose) for pose in palm_targets]),
        palm_position_error_m=metrics[:, 0],
        palm_orientation_error_rad=metrics[:, 1],
        foot_position_error_m=metrics[:, 2],
        foot_orientation_error_rad=metrics[:, 3],
        com_offset_m=metrics[:, 4],
        foot_target_names=np.asarray(FOOT_NAMES),
        foot_target_poses_wxyz=np.repeat(np.asarray([[pose_to_wxyz(foot_targets[name]) for name in FOOT_NAMES]]), frame_count, axis=0),
        support_mode=np.asarray("fixed_double_support"),
        optimizer_success=np.asarray(result.success),
        optimizer_cost=np.asarray(result.cost),
        optimizer_message=np.asarray(str(result.message)),
        model_version=np.asarray("full_horizon_quasistatic_v2"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **payload)
    print(f"success={result.success} cost={result.cost:.6g} nodes={frame_count}")
    print(f"max palm error={1000*np.max(metrics[:,0]):.3f} mm")
    print(f"max fixed-foot error={1000*np.max(metrics[:,2]):.3f} mm")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
