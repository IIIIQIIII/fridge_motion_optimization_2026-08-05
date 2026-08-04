#!/usr/bin/env python3
"""Refine the thumb pose per keyframe using live Isaac contact feedback."""

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
parser.add_argument("--report", required=True)
parser.add_argument("--samples", type=int, default=120)
parser.add_argument("--finger", choices=["thumb", "index", "middle", "ring", "pinky"], default="thumb")
parser.add_argument("--minimum-force", type=float, default=2.0)
parser.add_argument("--target-force", type=float, default=70.0)
parser.add_argument("--maximum-force", type=float, default=350.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
import numpy as np
import torch

import coordex.tasks.locomanip  # noqa: F401
from coordex.tasks.locomanip.fridge_env_cfg import FRIDGE_DOOR_JOINT
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse


FINGER_NUMBER = {"thumb": 1, "index": 2, "middle": 3, "ring": 4, "pinky": 5}
CANONICAL_BY_FINGER = {
    "thumb": np.array([0.6979, -0.1387, 1.5623, 0.7587]),
    "index": np.array([0.9907, 0.0764, 0.5865, 0.6327]),
    "middle": np.array([1.1013, 0.0250, 0.4591, 0.5109]),
    "ring": np.array([1.0113, 0.0297, 0.4939, 0.5544]),
    "pinky": np.array([0.8825, 0.0484, 0.5146, 0.5937]),
}


def main() -> None:
    source = np.load(args.trajectory)
    arrays = {key: source[key].copy() for key in source.files}
    joint_names = [str(name) for name in arrays["robot_joint_names"]]
    finger_number = FINGER_NUMBER[args.finger]
    finger_names = [f"right_finger{finger_number}_joint{j}" for j in range(1, 5)]
    canonical = CANONICAL_BY_FINGER[args.finger]
    finger_joint_ids = [joint_names.index(name) for name in finger_names]
    scene_reference = json.loads(Path(args.scene_reference).read_text())
    geometry = json.loads(Path(args.handle_geometry).read_text())
    upper = next(item for item in geometry if item["path"].endswith("/E_handle_1_7") and item["type"] == "Xform")

    cfg = parse_env_cfg("CoorDex-Fridge-Wuji-v0", device=args.device, num_envs=1)
    env = gym.make("CoorDex-Fridge-Wuji-v0", cfg=cfg)
    env.reset(seed=0)
    base = env.unwrapped
    robot = base.scene["robot"]
    fridge = base.scene["fridge"]
    command = base.command_manager.get_term("stage")
    sensor = base.scene.sensors[f"contact_force_{args.finger}"]
    device = base.device

    door_ids, _ = fridge.find_joints([FRIDGE_DOOR_JOINT], preserve_order=True)
    door_id = int(door_ids[0])
    tip_ids, _ = robot.find_bodies([f"right_finger{finger_number}_tip_link"], preserve_order=True)
    tip_id = int(tip_ids[0])
    sim_finger_ids = [robot.joint_names.index(name) for name in finger_names]
    limits = robot.data.soft_joint_pos_limits[0, sim_finger_ids].detach().cpu().numpy()

    door0_pos = torch.tensor(scene_reference["door_body_poses_wxyz"][0][:3], device=device)
    door0_quat = torch.tensor(scene_reference["door_body_poses_wxyz"][0][3:], device=device)
    center0 = torch.tensor(upper["center_w"], device=device)
    center_local = quat_apply_inverse(door0_quat.unsqueeze(0), (center0 - door0_pos).unsqueeze(0))[0]
    fridge_root = torch.tensor(scene_reference["fridge_root_pose_wxyz"], device=device).unsqueeze(0)
    rng = np.random.default_rng(7)
    report = []

    for frame in range(len(arrays["door_angle_rad"])):
        original = arrays["robot_joint_positions"][frame].copy()
        current = original[finger_joint_ids].copy()
        candidates = [current, canonical]
        candidates.extend((1.0 - u) * current + u * canonical for u in np.linspace(0.1, 0.9, 9))
        scale = np.array([0.38, 0.32, 0.34, 0.34])
        for center in (current, canonical):
            samples = center + rng.normal(size=(args.samples // 2, 4)) * scale
            candidates.extend(samples)
        candidates = np.clip(np.asarray(candidates), limits[:, 0], limits[:, 1])

        best = None
        attempts = []
        for candidate in candidates:
            q_np = original.copy()
            q_np[finger_joint_ids] = candidate
            root = torch.tensor(arrays["robot_root_pose_wxyz"][frame], device=device).unsqueeze(0)
            q = torch.tensor(q_np, device=device).unsqueeze(0)
            robot.write_root_pose_to_sim(root)
            robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            robot.set_joint_position_target(q)

            fridge_q = fridge.data.joint_pos.clone()
            fridge_q[:, door_id] = float(arrays["door_angle_rad"][frame])
            fridge.write_root_pose_to_sim(fridge_root)
            fridge.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
            fridge.write_joint_state_to_sim(fridge_q, torch.zeros_like(fridge_q))
            fridge.set_joint_position_target(fridge_q)
            base.scene.write_data_to_sim()
            base.sim.step(render=False)
            base.scene.update(base.sim.get_physics_dt())

            force = float(torch.linalg.norm(sensor.data.net_forces_w[0].reshape(-1, 3), dim=-1).max().item())
            door_pos = fridge.data.body_pos_w[0, command.door_body_id]
            door_quat = fridge.data.body_quat_w[0, command.door_body_id]
            handle_center = door_pos + quat_apply(door_quat.unsqueeze(0), center_local.unsqueeze(0))[0]
            tip = robot.data.body_pos_w[0, tip_id]
            distance = float(torch.linalg.norm(tip - handle_center).item())

            # Require actual contact close to the handle.  Prefer moderate
            # forces and minimal movement from the kinematic surface solution.
            contact_penalty = 0.0 if force >= args.minimum_force else 100.0
            excessive_penalty = max(force - args.maximum_force, 0.0) / 25.0
            force_preference = abs(np.log((force + 1.0) / args.target_force))
            distance_penalty = max(distance - 0.042, 0.0) * 300.0
            motion_penalty = 0.18 * np.linalg.norm(candidate - current)
            score = contact_penalty + excessive_penalty + force_preference + distance_penalty + motion_penalty
            attempts.append({"force_n": force, "distance_m": distance, "score": float(score)})
            if best is None or score < best[0]:
                best = (score, candidate.copy(), force, distance)

        assert best is not None
        arrays["robot_joint_positions"][frame, finger_joint_ids] = best[1]
        report.append(
            {
                "frame": frame,
                "finger": args.finger,
                "selected_joint_pos": best[1].tolist(),
                "selected_force_n": float(best[2]),
                "selected_tip_distance_m": float(best[3]),
                "candidate_count": len(candidates),
                "contact_candidates": int(sum(item["force_n"] >= args.minimum_force for item in attempts)),
            }
        )
        print(
            f"frame={frame} finger={args.finger} force_n={best[2]:.2f} tip_distance_mm={1000*best[3]:.1f} "
            f"contact_candidates={report[-1]['contact_candidates']}/{len(candidates)}",
            flush=True,
        )

    np.savez(args.output, **arrays)
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"REFINED_TRAJECTORY={args.output}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
