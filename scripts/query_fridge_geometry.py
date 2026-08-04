#!/usr/bin/env python3
"""Inspect the rendered fridge geometry and report handle-like prim bounds.

Run inside the same Isaac Lab environment used for rendering.  The output is a
JSON inventory of prims whose path contains ``handle`` plus their world-space
axis-aligned bounds.  This intentionally queries geometry rather than reusing
the task command's single interaction point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import omni.usd
from pxr import Usd, UsdGeom

import coordex.tasks.locomanip  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    cfg = parse_env_cfg("CoorDex-Fridge-Wuji-v0", device=args.device, num_envs=1)
    env = gym.make("CoorDex-Fridge-Wuji-v0", cfg=cfg)
    env.reset()

    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    records = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "handle" not in path.lower():
            continue
        bound = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        lo = bound.GetMin()
        hi = bound.GetMax()
        records.append(
            {
                "path": path,
                "type": prim.GetTypeName(),
                "min_w": [float(lo[i]) for i in range(3)],
                "max_w": [float(hi[i]) for i in range(3)],
                "center_w": [float((lo[i] + hi[i]) * 0.5) for i in range(3)],
                "extent_w": [float(hi[i] - lo[i]) for i in range(3)],
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2))
    print(f"HANDLE_GEOMETRY={output}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
