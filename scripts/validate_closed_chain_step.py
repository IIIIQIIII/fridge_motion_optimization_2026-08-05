#!/usr/bin/env python3
"""Independently validate a DS->SS->DS task-space stepping reference."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from math_model.centroidal import solve_centroidal_equilibrium
from math_model.quasistatic import FootPatch
from math_model.stepping import SupportMode

DEFAULT_INPUT = ROOT / "outputs/sonic_closed_chain_step_reference.npz"
DEFAULT_OUTPUT = ROOT / "outputs/sonic_closed_chain_step_validation.json"


def rotation_z(yaw: float) -> np.ndarray:
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    data = np.load(args.input)
    time_s = np.asarray(data["time_s"], dtype=float)
    modes = np.asarray(data["support_mode"]).astype(str)
    swing_foot = str(data["swing_foot"])
    left_positions = np.asarray(data["left_foot_target_position_w"])
    right_positions = np.asarray(data["right_foot_target_position_w"])
    left_yaw = np.asarray(data["left_foot_target_yaw_rad"])
    right_yaw = np.asarray(data["right_foot_target_yaw_rad"])
    com = np.asarray(data["com_target_w"])
    com_acceleration = np.gradient(
        np.gradient(com, time_s, axis=0, edge_order=2),
        time_s,
        axis=0,
        edge_order=2,
    )
    hand = np.asarray(data["handle_position_w"])
    hand_force = np.asarray(data["hand_force_on_robot_w"])
    shoulder = np.asarray(data["right_shoulder_reference_w"])
    arm_reach = np.linalg.norm(hand - shoulder, axis=1)
    mass = float(data["robot_mass_kg"])
    friction = float(data["friction_coefficient"])
    half_length = float(data["foot_half_length_m"])
    half_width = float(data["foot_half_width_m"])
    expected_clearance = float(data["swing_clearance_m"])

    single_support = np.flatnonzero(modes != SupportMode.DOUBLE_SUPPORT.value)
    sequence_ok = bool(
        len(single_support) > 0
        and np.all(np.diff(single_support) == 1)
        and np.all(
            modes[: single_support[0]] == SupportMode.DOUBLE_SUPPORT.value
        )
        and np.all(
            modes[single_support[-1] + 1 :] == SupportMode.DOUBLE_SUPPORT.value
        )
    )

    stance_positions = left_positions if swing_foot == "right" else right_positions
    swing_positions = right_positions if swing_foot == "right" else left_positions
    stance_drift = float(
        np.max(np.linalg.norm(stance_positions - stance_positions[0], axis=1))
    )
    clearance = float(np.max(swing_positions[:, 2] - swing_positions[0, 2]))

    contact_results = []
    for frame, mode in enumerate(modes):
        active_feet = []
        if mode in {
            SupportMode.DOUBLE_SUPPORT.value,
            SupportMode.LEFT_SUPPORT.value,
        }:
            active_feet.append(
                FootPatch(
                    left_positions[frame],
                    rotation_z(float(left_yaw[frame])),
                    half_length,
                    half_width,
                    "left",
                )
            )
        if mode in {
            SupportMode.DOUBLE_SUPPORT.value,
            SupportMode.RIGHT_SUPPORT.value,
        }:
            active_feet.append(
                FootPatch(
                    right_positions[frame],
                    rotation_z(float(right_yaw[frame])),
                    half_length,
                    half_width,
                    "right",
                )
            )
        contact_results.append(
            solve_centroidal_equilibrium(
                mass=mass,
                center_of_mass=com[frame],
                com_acceleration=com_acceleration[frame],
                angular_momentum_rate=np.zeros(3),
                hand_point=hand[frame],
                hand_force_on_robot=hand_force[frame],
                feet=active_feet,
                friction_coefficient=friction,
            )
        )

    feasible_fraction = float(
        np.mean([result.feasible for result in contact_results])
    )
    minimum_support = float(
        min(result.support_margin for result in contact_results)
    )
    minimum_friction = float(
        min(result.minimum_friction_margin for result in contact_results)
    )
    tests = {
        "contact_sequence_ds_ss_ds": {
            "observed": sequence_ok,
            "passed": sequence_ok,
        },
        "stance_foot_drift_m": {
            "observed": stance_drift,
            "threshold": 1e-6,
            "passed": stance_drift <= 1e-6,
        },
        "peak_swing_clearance_m": {
            "observed": clearance,
            "threshold": 0.95 * expected_clearance,
            "passed": clearance >= 0.95 * expected_clearance,
        },
        "centroidal_contact_feasible_fraction": {
            "observed": feasible_fraction,
            "threshold": 1.0,
            "passed": feasible_fraction == 1.0,
        },
        "minimum_support_margin_m": {
            "observed": minimum_support,
            "threshold": 0.0,
            "passed": minimum_support >= -1e-6,
        },
        "minimum_friction_margin_n": {
            "observed": minimum_friction,
            "threshold": 0.0,
            "passed": minimum_friction >= -1e-5,
        },
        "maximum_arm_reach_m": {
            "observed": float(np.max(arm_reach)),
            "threshold": 0.42,
            "passed": float(np.max(arm_reach)) <= 0.42,
        },
    }
    accepted = all(test["passed"] for test in tests.values())
    report = {
        "accepted": accepted,
        "model": "closed_chain_ds_ss_ds_v3_independent_validation",
        "input": str(args.input),
        "tests": tests,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"accepted={accepted}")
    for name, test in tests.items():
        print(
            f"{'PASS' if test['passed'] else 'FAIL'} {name}: "
            f"{test['observed']}"
        )
    print(f"Wrote {args.output}")
    if args.fail_on_error and not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
