#!/usr/bin/env python3
"""Render mathematically optimized G1 keyframes with Isaac Sim RTX cameras."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--trajectory", required=True)
parser.add_argument("--scene-reference", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--interpolation-frames", type=int, default=25)
parser.add_argument("--camera-eye", nargs=3, type=float, default=[3.15, -2.75, 1.72])
parser.add_argument("--camera-target", nargs=3, type=float, default=[1.10, 0.02, 0.78])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
import coordex.tasks.locomanip  # noqa: F401
import isaaclab.sim as sim_utils
from coordex.tasks.locomanip.fridge_env_cfg import FRIDGE_DOOR_JOINT
from isaaclab.sensors.camera import CameraCfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix


def minimum_jerk(u: float) -> float:
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def interpolate_trajectory(data, count: int):
    key_count = len(data["door_angle_rad"])
    sample_positions = np.linspace(0.0, key_count - 1, count)
    root_out, joints_out, door_out = [], [], []
    for position in sample_positions:
        left = min(int(np.floor(position)), key_count - 2)
        u = minimum_jerk(float(position - left))
        root_a = data["robot_root_pose_wxyz"][left]
        root_b = data["robot_root_pose_wxyz"][left + 1]
        root = (1.0 - u) * root_a + u * root_b
        root[3:] /= np.linalg.norm(root[3:])
        root_out.append(root)
        joints_out.append((1.0 - u) * data["robot_joint_positions"][left] + u * data["robot_joint_positions"][left + 1])
        door_out.append((1.0 - u) * data["door_angle_rad"][left] + u * data["door_angle_rad"][left + 1])
    return np.asarray(root_out), np.asarray(joints_out), np.asarray(door_out)


def main():
    trajectory = np.load(args.trajectory)
    scene_reference = json.loads(Path(args.scene_reference).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = parse_env_cfg("CoorDex-Fridge-Wuji-v0", device=args.device, num_envs=1)
    eye = torch.tensor([args.camera_eye], device=args.device, dtype=torch.float32)
    target = torch.tensor([args.camera_target], device=args.device, dtype=torch.float32)
    camera_quat = quat_from_matrix(create_rotation_matrix_from_view(eye, target, "Z", device=args.device))[0]
    cfg.scene.math_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/MathRenderCamera",
        update_period=0.0,
        height=720,
        width=960,
        data_types=["rgb"],
        offset=CameraCfg.OffsetCfg(
            pos=tuple(float(value) for value in args.camera_eye),
            rot=tuple(float(value) for value in camera_quat.detach().cpu().tolist()),
            convention="opengl",
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=24.0,
            clipping_range=(0.05, 100.0),
        ),
    )
    # Isaac Sim 5.1 may leave this global flag true after all scene assets are
    # usable.  The camera is static, so reset need not wait on that flag.
    cfg.wait_for_textures = False
    env = gym.make("CoorDex-Fridge-Wuji-v0", cfg=cfg)
    env.reset(seed=0)
    base = env.unwrapped
    print("RENDER_STAGE=environment_ready", flush=True)
    robot = base.scene["robot"]
    fridge = base.scene["fridge"]

    expected_names = [str(name) for name in trajectory["robot_joint_names"]]
    if list(robot.joint_names) != expected_names:
        raise RuntimeError("Isaac joint order differs from exported trajectory")
    door_ids, found = fridge.find_joints([FRIDGE_DOOR_JOINT], preserve_order=True)
    if list(found) != [FRIDGE_DOOR_JOINT]:
        raise RuntimeError(f"Door joint not found: {found}")
    door_id = int(door_ids[0])

    camera = base.scene["math_camera"]
    print(f"RENDER_CAMERA_POS={[args.camera_eye]}", flush=True)
    print(f"RENDER_CAMERA_QUAT={camera_quat.detach().cpu().tolist()}", flush=True)
    print("RENDER_STAGE=camera_pose_ready", flush=True)

    roots, joints, doors = interpolate_trajectory(trajectory, args.interpolation_frames)
    key_indices = np.round(np.linspace(0, args.interpolation_frames - 1, 5)).astype(int)
    frames = []
    keyframes = []

    for frame_index, (root_pose_np, joint_pos_np, door_angle) in enumerate(zip(roots, joints, doors)):
        print(f"RENDER_STAGE=frame_{frame_index:03d}_state", flush=True)
        root_pose = torch.tensor(root_pose_np, device=base.device, dtype=torch.float32).unsqueeze(0)
        joint_pos = torch.tensor(joint_pos_np, device=base.device, dtype=torch.float32).unsqueeze(0)
        joint_vel = torch.zeros_like(joint_pos)
        robot.write_root_pose_to_sim(root_pose)
        robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=base.device))
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.set_joint_position_target(joint_pos)
        robot.set_joint_velocity_target(joint_vel)

        fridge_pos = fridge.data.joint_pos.clone()
        fridge_pos[:, door_id] = float(door_angle)
        fridge_vel = torch.zeros_like(fridge_pos)
        fridge_root_pose = torch.tensor(
            scene_reference["fridge_root_pose_wxyz"], device=base.device, dtype=torch.float32
        ).unsqueeze(0)
        fridge.write_root_pose_to_sim(fridge_root_pose)
        fridge.write_root_velocity_to_sim(torch.zeros((1, 6), device=base.device))
        fridge.write_joint_state_to_sim(fridge_pos, fridge_vel)
        fridge.set_joint_position_target(fridge_pos)
        fridge.set_joint_velocity_target(fridge_vel)
        base.scene.write_data_to_sim()
        print(
            f"RENDER_ROOTS=robot:{robot.data.root_pos_w.detach().cpu().numpy().tolist()} "
            f"fridge:{fridge.data.root_pos_w.detach().cpu().numpy().tolist()}",
            flush=True,
        )
        print(f"RENDER_STAGE=frame_{frame_index:03d}_sim_render", flush=True)
        # Advance one simulation/render tick so the static RTX render product
        # publishes a fresh image instead of returning its frame-zero cache.
        base.sim.step(render=True)
        base.scene.update(base.sim.get_physics_dt())
        print(f"RENDER_STAGE=frame_{frame_index:03d}_camera_update", flush=True)
        camera.update(base.sim.get_physics_dt(), force_recompute=True)
        print(f"RENDER_STAGE=frame_{frame_index:03d}_readback", flush=True)
        rgb = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
        frames.append(rgb)
        imageio.imwrite(output_dir / f"isaac_frame_{frame_index:03d}.png", rgb)
        if frame_index in key_indices:
            keyframes.append(rgb)

    imageio.mimsave(output_dir / "g1_fridge_isaac.gif", frames + frames[-2:0:-1], duration=0.12, loop=0)
    imageio.mimsave(output_dir / "g1_fridge_isaac.mp4", frames, fps=12, codec="libx264", quality=8)

    separator = np.full((720, 4, 3), 235, dtype=np.uint8)
    composite = keyframes[0]
    for frame in keyframes[1:]:
        composite = np.concatenate([composite, separator, frame], axis=1)
    imageio.imwrite(output_dir / "g1_fridge_isaac_keyframes.png", composite)
    print(f"ISAAC_RENDER_OUTPUT={output_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
