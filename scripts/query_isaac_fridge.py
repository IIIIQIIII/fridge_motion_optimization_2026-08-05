#!/usr/bin/env python3
"""Export the real Isaac fridge handle arc and pre-grasp robot state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
import torch

import coordex.tasks.locomanip  # noqa: F401
from coordex.tasks.locomanip.fridge_env_cfg import FRIDGE_DOOR_JOINT
from isaaclab_tasks.utils import parse_env_cfg


def main():
    cfg = parse_env_cfg("CoorDex-Fridge-Wuji-v0", device=args.device, num_envs=1)
    env = gym.make("CoorDex-Fridge-Wuji-v0", cfg=cfg)
    env.reset()
    base = env.unwrapped
    robot = base.scene["robot"]
    fridge = base.scene["fridge"]
    command = base.command_manager.get_term("stage")
    door_ids, _ = fridge.find_joints([FRIDGE_DOOR_JOINT], preserve_order=True)
    door_id = int(door_ids[0])

    angles_deg = [0, 15, 30, 45, 60]
    handles = []
    door_poses = []
    for degrees in angles_deg:
        q = fridge.data.joint_pos.clone()
        q[:, door_id] = torch.deg2rad(torch.tensor(float(degrees), device=base.device))
        v = torch.zeros_like(q)
        fridge.write_joint_state_to_sim(q, v)
        fridge.set_joint_position_target(q)
        fridge.set_joint_velocity_target(v)
        fridge.write_data_to_sim()
        for _ in range(3):
            base.sim.step(render=False)
            base.scene.update(base.sim.get_physics_dt())
        handles.append(command.handle_pos_w()[0].detach().cpu().tolist())
        door_poses.append(
            torch.cat(
                [
                    fridge.data.body_pos_w[0, command.door_body_id],
                    fridge.data.body_quat_w[0, command.door_body_id],
                ]
            ).detach().cpu().tolist()
        )

    payload = {
        "angles_deg": angles_deg,
        "handle_positions_w": handles,
        "door_body_poses_wxyz": door_poses,
        "robot_root_pose_wxyz": robot.data.root_pose_w[0].detach().cpu().tolist(),
        "robot_joint_names": list(robot.joint_names),
        "robot_joint_positions": robot.data.joint_pos[0].detach().cpu().tolist(),
        "fridge_root_pose_wxyz": fridge.data.root_pose_w[0].detach().cpu().tolist(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(f"SCENE_REFERENCE={output}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
