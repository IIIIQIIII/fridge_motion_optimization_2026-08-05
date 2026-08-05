#!/usr/bin/env python3
"""Run the coupled fixed-support solve and independent physics audit."""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/isaac_wholebody_power_grasp_action.npz")
    parser.add_argument("--trajectory", type=Path, default=ROOT / "outputs/isaac_full_horizon_fixed_support_action.npz")
    parser.add_argument("--audit", type=Path, default=ROOT / "outputs/full_horizon_quasistatic_audit.json")
    parser.add_argument("--nodes", type=int, default=13)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--fail-on-step-required", action="store_true")
    args = parser.parse_args()
    subprocess.run([
        sys.executable, str(ROOT / "scripts/solve_full_horizon_quasistatic.py"),
        "--input", str(args.input), "--output", str(args.trajectory),
        "--nodes", str(args.nodes), "--duration", str(args.duration),
    ], check=True, cwd=ROOT)
    command = [
        sys.executable, str(ROOT / "scripts/audit_quasistatic_trajectory.py"),
        "--input", str(args.trajectory), "--output", str(args.audit),
    ]
    if args.fail_on_step_required:
        command.append("--fail-on-step-required")
    subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
