#!/usr/bin/env python3
"""Run the closed-chain stepping planner and its independent validator."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "outputs/isaac_full_horizon_fixed_support_action.npz",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=ROOT / "outputs/full_horizon_quasistatic_audit.json",
    )
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--fail-on-infeasible", action="store_true")
    args = parser.parse_args()

    reference = ROOT / "outputs/sonic_closed_chain_step_reference.npz"
    report = ROOT / "outputs/sonic_closed_chain_step_report.json"
    validation = ROOT / "outputs/sonic_closed_chain_step_validation.json"

    command = [
        sys.executable,
        str(ROOT / "scripts/plan_closed_chain_step.py"),
        "--input",
        str(args.input),
        "--audit",
        str(args.audit),
        "--output",
        str(reference),
        "--report",
        str(report),
        "--duration",
        str(args.duration),
    ]
    if args.fail_on_infeasible:
        command.append("--fail-on-infeasible")
    subprocess.run(command, check=True)

    command = [
        sys.executable,
        str(ROOT / "scripts/validate_closed_chain_step.py"),
        "--input",
        str(reference),
        "--output",
        str(validation),
    ]
    if args.fail_on_infeasible:
        command.append("--fail-on-error")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
