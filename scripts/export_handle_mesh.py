#!/usr/bin/env python3
"""Export the real upper-handle render mesh in world coordinates.

This is intentionally separate from the coarse Xform AABB inventory: the
handle Xform contains the vertical grip bar and its mounting brackets, so its
overall box is not the cross-section that the hand actually wraps around.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
import numpy as np
import omni.usd
from pxr import Usd, UsdGeom

import coordex.tasks.locomanip  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


UPPER_HANDLE = "/World/envs/env_0/Fridge/E_door_1/E_handle_1_7"


def triangulate(counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    triangles: list[list[int]] = []
    offset = 0
    for count in counts:
        face = indices[offset : offset + int(count)]
        for i in range(1, len(face) - 1):
            triangles.append([int(face[0]), int(face[i]), int(face[i + 1])])
        offset += int(count)
    return np.asarray(triangles, dtype=np.int32)


def main() -> None:
    cfg = parse_env_cfg("CoorDex-Fridge-Wuji-v0", device=args.device, num_envs=1)
    env = gym.make("CoorDex-Fridge-Wuji-v0", cfg=cfg)
    env.reset(seed=0)
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    points_parts: list[np.ndarray] = []
    triangles_parts: list[np.ndarray] = []
    paths: list[str] = []
    vertex_offset = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(UPPER_HANDLE) or not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        world = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64)
        # USD/Gf uses row-vector multiplication in Python.
        points_h = np.c_[points, np.ones(len(points))]
        points_w = (points_h @ world)[:, :3]
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        triangles = triangulate(counts, indices) + vertex_offset
        points_parts.append(points_w)
        triangles_parts.append(triangles)
        paths.append(path)
        vertex_offset += len(points_w)

    if not points_parts:
        raise RuntimeError(f"No mesh found below {UPPER_HANDLE}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        vertices_w=np.concatenate(points_parts),
        triangles=np.concatenate(triangles_parts),
        mesh_paths=np.asarray(paths),
    )
    vertices = np.concatenate(points_parts)
    print(f"HANDLE_MESH={output}")
    print(f"mesh_prims={len(paths)} vertices={len(vertices)} triangles={sum(map(len, triangles_parts))}")
    print(f"bounds_min={vertices.min(axis=0).tolist()}")
    print(f"bounds_max={vertices.max(axis=0).tolist()}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
