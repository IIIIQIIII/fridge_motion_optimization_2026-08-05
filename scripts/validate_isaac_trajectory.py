#!/usr/bin/env python3
"""Validate a fridge trajectory against Isaac body geometry and contacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--trajectory", required=True)
parser.add_argument("--scene-reference", required=True)
parser.add_argument("--handle-geometry", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--interpolation-frames", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
import numpy as np
import torch

import coordex.tasks.locomanip  # noqa: F401
from coordex.tasks.locomanip.constants import RIGHT_HAND_TIP_NAMES
from coordex.tasks.locomanip.fridge_env_cfg import FRIDGE_DOOR_JOINT
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_error_magnitude


def main() -> None:
    trajectory = np.load(args.trajectory)
    scene_reference = json.loads(Path(args.scene_reference).read_text())
    geometry = json.loads(Path(args.handle_geometry).read_text())
    upper = next(item for item in geometry if item["path"].endswith("/E_handle_1_7") and item["type"] == "Xform")
    handle_center0 = torch.tensor(upper["center_w"], dtype=torch.float32)

    cfg = parse_env_cfg("CoorDex-Fridge-Wuji-v0", device=args.device, num_envs=1)
    env = gym.make("CoorDex-Fridge-Wuji-v0", cfg=cfg)
    env.reset(seed=0)
    base = env.unwrapped
    robot = base.scene["robot"]
    fridge = base.scene["fridge"]
    command = base.command_manager.get_term("stage")
    device = base.device

    expected_names = [str(x) for x in trajectory["robot_joint_names"]]
    if list(robot.joint_names) != expected_names:
        raise RuntimeError("Isaac joint order differs from trajectory")
    door_ids, _ = fridge.find_joints([FRIDGE_DOOR_JOINT], preserve_order=True)
    door_id = int(door_ids[0])
    body_queries = ["right_palm_link", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link", *RIGHT_HAND_TIP_NAMES]
    body_ids, found = robot.find_bodies(body_queries, preserve_order=True)
    if list(found) != body_queries:
        raise RuntimeError(f"Missing robot bodies: requested={body_queries}, found={found}")
    body_id = dict(zip(body_queries, [int(x) for x in body_ids]))

    # Express the mesh center in the real door-body frame at the zero-angle
    # state, then use the live Isaac door transform for every tested frame.
    door0_pos = torch.tensor(scene_reference["door_body_poses_wxyz"][0][:3], device=device)
    door0_quat = torch.tensor(scene_reference["door_body_poses_wxyz"][0][3:], device=device)
    handle_center0 = handle_center0.to(device)
    handle_local = quat_apply_inverse(door0_quat.unsqueeze(0), (handle_center0 - door0_pos).unsqueeze(0))[0]

    if args.interpolation_frames and args.interpolation_frames > len(trajectory["door_angle_rad"]):
        sample_positions = np.linspace(0.0, len(trajectory["door_angle_rad"]) - 1, args.interpolation_frames)
        roots, joints, doors = [], [], []
        for position in sample_positions:
            left = min(int(np.floor(position)), len(trajectory["door_angle_rad"]) - 2)
            raw = float(position - left)
            u = 10.0 * raw**3 - 15.0 * raw**4 + 6.0 * raw**5
            root = (1.0 - u) * trajectory["robot_root_pose_wxyz"][left] + u * trajectory["robot_root_pose_wxyz"][left + 1]
            root[3:] /= np.linalg.norm(root[3:])
            roots.append(root)
            joints.append((1.0 - u) * trajectory["robot_joint_positions"][left] + u * trajectory["robot_joint_positions"][left + 1])
            doors.append((1.0 - u) * trajectory["door_angle_rad"][left] + u * trajectory["door_angle_rad"][left + 1])
        roots = np.asarray(roots)
        joints = np.asarray(joints)
        doors = np.asarray(doors)
    else:
        roots = trajectory["robot_root_pose_wxyz"]
        joints = trajectory["robot_joint_positions"]
        doors = trajectory["door_angle_rad"]

    records = []
    initial_torso_quat = None
    names = expected_names
    for frame_index in range(len(doors)):
        root_pose = torch.tensor(roots[frame_index], device=device).unsqueeze(0)
        joint_pos = torch.tensor(joints[frame_index], device=device).unsqueeze(0)
        zeros = torch.zeros_like(joint_pos)
        robot.write_root_pose_to_sim(root_pose)
        robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
        robot.write_joint_state_to_sim(joint_pos, zeros)
        robot.set_joint_position_target(joint_pos)
        robot.set_joint_velocity_target(zeros)

        fridge_pos = fridge.data.joint_pos.clone()
        fridge_pos[:, door_id] = float(doors[frame_index])
        fridge_root = torch.tensor(scene_reference["fridge_root_pose_wxyz"], device=device).unsqueeze(0)
        fridge.write_root_pose_to_sim(fridge_root)
        fridge.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
        fridge.write_joint_state_to_sim(fridge_pos, torch.zeros_like(fridge_pos))
        fridge.set_joint_position_target(fridge_pos)
        base.scene.write_data_to_sim()
        base.sim.step(render=False)
        base.scene.update(base.sim.get_physics_dt())

        door_pos = fridge.data.body_pos_w[0, command.door_body_id]
        door_quat = fridge.data.body_quat_w[0, command.door_body_id]
        handle_center = door_pos + quat_apply(door_quat.unsqueeze(0), handle_local.unsqueeze(0))[0]
        palm_pos = robot.data.body_pos_w[0, body_id["right_palm_link"]]
        palm_quat = robot.data.body_quat_w[0, body_id["right_palm_link"]]
        handle_in_palm = quat_apply_inverse(palm_quat.unsqueeze(0), (handle_center - palm_pos).unsqueeze(0))[0]
        tips = torch.stack([robot.data.body_pos_w[0, body_id[name]] for name in RIGHT_HAND_TIP_NAMES])
        tips_in_palm = quat_apply_inverse(
            palm_quat.unsqueeze(0).expand(len(RIGHT_HAND_TIP_NAMES), -1), tips - palm_pos
        )
        tip_to_center = torch.linalg.norm(tips - handle_center, dim=-1)
        tip_radial = torch.linalg.norm((tips - handle_center)[:, :2], dim=-1)
        torso_quat = robot.data.body_quat_w[0, body_id["torso_link"]]
        if initial_torso_quat is None:
            initial_torso_quat = torso_quat.clone()
        torso_delta = quat_error_magnitude(initial_torso_quat.unsqueeze(0), torso_quat.unsqueeze(0))[0]

        try:
            fingertip_contact = bool(command.fingertip_contact()[0].item())
        except Exception:
            fingertip_contact = None
        per_finger_force = {}
        for sensor_name in (
            "contact_force_thumb",
            "contact_force_index",
            "contact_force_middle",
            "contact_force_ring",
            "contact_force_pinky",
        ):
            sensor = base.scene.sensors[sensor_name]
            force = sensor.data.net_forces_w[0]
            per_finger_force[sensor_name.removeprefix("contact_force_")] = float(
                torch.linalg.norm(force.reshape(-1, 3), dim=-1).max().item()
            )

        angle = lambda name: float(np.rad2deg(joints[frame_index, names.index(name)]))
        records.append(
            {
                "frame": frame_index,
                "door_angle_deg": float(np.rad2deg(doors[frame_index])),
                "handle_center_w": handle_center.detach().cpu().tolist(),
                "handle_in_palm": handle_in_palm.detach().cpu().tolist(),
                "tip_positions_in_palm": tips_in_palm.detach().cpu().tolist(),
                "tip_to_handle_center_m": tip_to_center.detach().cpu().tolist(),
                "tip_radial_to_handle_axis_m": tip_radial.detach().cpu().tolist(),
                "fingertip_contact": fingertip_contact,
                "per_finger_contact_force_n": per_finger_force,
                "torso_delta_deg": float(torch.rad2deg(torso_delta).item()),
                "root_height_m": float(root_pose[0, 2].item()),
                "waist_yaw_deg": angle("waist_yaw_joint"),
                "left_knee_deg": angle("left_knee_joint"),
                "right_knee_deg": angle("right_knee_joint"),
                "right_elbow_deg": angle("right_elbow_joint"),
                "right_wrist_roll_deg": angle("right_wrist_roll_joint"),
                "right_wrist_pitch_deg": angle("right_wrist_pitch_joint"),
                "right_wrist_yaw_deg": angle("right_wrist_yaw_joint"),
            }
        )

    output = {
        "handle_extent_w_at_zero": upper["extent_w"],
        "right_hand_tip_names": list(RIGHT_HAND_TIP_NAMES),
        "scene_sensors": list(base.scene.sensors.keys()),
        "frames": records,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2))
    print(f"ISAAC_VALIDATION={path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
