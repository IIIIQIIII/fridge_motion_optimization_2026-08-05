#!/usr/bin/env python3
"""Plan one DS->SS->DS step while the right hand stays on the handle.

Input is the fixed-support V2 trajectory. Output is a task-space reference for a
whole-body motion tracker: support mode, left/right foot targets, COM, pelvis,
and the original right-wrist closed-chain target. The planner selects which foot
moves, lift/landing timing, and landing pose, then audits centroidal balance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from math_model.quasistatic import (
    DoorResistanceModel,
    fit_hinge_circle,
    required_tangential_hand_force,
)
from math_model.stepping import (
    FootPose,
    StepPlannerConfig,
    StepPlanningProblem,
    plan_best_single_step,
)
from generate_action_keyframes import ROOT as PROJECT_ROOT, URDF_PATH, UrdfKinematics

DEFAULT_INPUT = PROJECT_ROOT / "outputs/isaac_full_horizon_fixed_support_action.npz"
DEFAULT_AUDIT = PROJECT_ROOT / "outputs/full_horizon_quasistatic_audit.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/sonic_closed_chain_step_reference.npz"
DEFAULT_REPORT = PROJECT_ROOT / "outputs/sonic_closed_chain_step_report.json"
FOOT_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")


def pose_from_wxyz(values: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = values[:3]
    pose[:3, :3] = Rotation.from_quat(values[[4, 5, 6, 3]]).as_matrix()
    return pose


def yaw_from_rotation(rotation: np.ndarray) -> float:
    return float(Rotation.from_matrix(rotation).as_euler("xyz")[2])


def trigger_from_audit(path: Path, angles: np.ndarray) -> int | None:
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    value = report.get("first_required_contact_mode_change_deg")
    if value is None:
        return None
    return int(np.argmin(np.abs(np.rad2deg(angles) - float(value))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--friction", type=float, default=0.7)
    parser.add_argument("--swing-clearance", type=float, default=0.065)
    parser.add_argument("--max-step-length", type=float, default=0.34)
    parser.add_argument("--max-arm-reach", type=float, default=0.42)
    parser.add_argument("--breakaway-torque", type=float, default=6.0)
    parser.add_argument("--coulomb-torque", type=float, default=3.0)
    parser.add_argument("--spring-torque-per-rad", type=float, default=0.5)
    parser.add_argument("--fail-on-infeasible", action="store_true")
    args = parser.parse_args()
    if args.duration <= 0.0:
        raise ValueError("duration must be positive")

    data = np.load(args.input)
    required = {
        "door_angle_rad",
        "handle_positions_w",
        "palm_target_poses_wxyz",
        "robot_root_pose_wxyz",
        "robot_joint_names",
        "robot_joint_positions",
    }
    missing = sorted(required.difference(data.files))
    if missing:
        raise KeyError(f"trajectory is missing: {missing}")

    angles = np.asarray(data["door_angle_rad"], dtype=float)
    frame_count = len(angles)
    time_s = np.linspace(0.0, args.duration, frame_count)
    handles = np.asarray(data["handle_positions_w"], dtype=float)
    names = [str(name) for name in data["robot_joint_names"]]
    model = UrdfKinematics(URDF_PATH)
    mass = float(sum(value[0] for value in model.inertials.values()))

    pelvis_positions = []
    pelvis_yaw = []
    shoulder_positions = []
    center_of_mass = []
    initial_feet: dict[str, FootPose] | None = None
    for root_values, joint_values in zip(
        data["robot_root_pose_wxyz"], data["robot_joint_positions"]
    ):
        base = pose_from_wxyz(root_values)
        joints = dict(zip(names, joint_values))
        poses = model.forward(base, joints)
        pelvis_positions.append(base[:3, 3])
        pelvis_yaw.append(yaw_from_rotation(base[:3, :3]))
        shoulder_positions.append(poses["right_shoulder_roll_link"][:3, 3])
        center_of_mass.append(model.center_of_mass(poses))
        if initial_feet is None:
            initial_feet = {}
            for foot_name, label in zip(FOOT_NAMES, ("left", "right")):
                foot = poses[foot_name]
                sole_center = foot[:3, 3] + foot[:3, :3] @ np.array(
                    [0.0, 0.0, -0.035]
                )
                initial_feet[label] = FootPose(
                    sole_center,
                    yaw_from_rotation(foot[:3, :3]),
                    label,
                )
    assert initial_feet is not None

    hinge = fit_hinge_circle(handles)
    angular_speed = np.gradient(angles, time_s)
    door = DoorResistanceModel(
        args.breakaway_torque,
        args.coulomb_torque,
        args.spring_torque_per_rad,
    )
    hand_forces = np.vstack(
        [
            required_tangential_hand_force(
                hinge.center,
                hinge.axis,
                handle,
                door.torque(angle, speed),
            )
            for handle, angle, speed in zip(handles, angles, angular_speed)
        ]
    )

    problem = StepPlanningProblem(
        time_s,
        angles,
        handles,
        hand_forces,
        np.asarray(pelvis_positions),
        np.asarray(pelvis_yaw),
        np.asarray(shoulder_positions),
        np.asarray(center_of_mass),
        initial_feet["left"],
        initial_feet["right"],
        mass,
    )
    config = StepPlannerConfig(
        swing_clearance_m=args.swing_clearance,
        friction_coefficient=args.friction,
        max_step_length_m=args.max_step_length,
        max_arm_reach_m=args.max_arm_reach,
    )
    trigger = trigger_from_audit(args.audit, angles)
    plan = plan_best_single_step(problem, config, trigger_index=trigger)

    contact_feasible = np.asarray(
        [result.feasible for result in plan.contact_results], dtype=bool
    )
    support_margin = np.asarray(
        [result.support_margin for result in plan.contact_results], dtype=float
    )
    friction_margin = np.asarray(
        [result.minimum_friction_margin for result in plan.contact_results],
        dtype=float,
    )
    payload = {
        "time_s": time_s,
        "door_angle_rad": angles,
        "support_mode": plan.support_modes,
        "swing_foot": np.asarray(plan.candidate.swing_foot),
        "lift_index": np.asarray(plan.candidate.lift_index),
        "land_index": np.asarray(plan.candidate.land_index),
        "left_foot_target_position_w": plan.left_foot_positions_w,
        "right_foot_target_position_w": plan.right_foot_positions_w,
        "left_foot_target_yaw_rad": plan.left_foot_yaw_rad,
        "right_foot_target_yaw_rad": plan.right_foot_yaw_rad,
        "com_target_w": plan.com_positions_w,
        "com_acceleration_target_w": plan.com_acceleration_w,
        "pelvis_target_position_w": plan.pelvis_positions_w,
        "pelvis_target_yaw_rad": plan.pelvis_yaw_rad,
        "right_shoulder_reference_w": plan.shoulder_positions_w,
        "right_wrist_target_poses_wxyz": data["palm_target_poses_wxyz"],
        "handle_position_w": handles,
        "hand_force_on_robot_w": hand_forces,
        "arm_reach_m": plan.arm_reach_m,
        "contact_feasible": contact_feasible,
        "support_margin_m": support_margin,
        "friction_margin_n": friction_margin,
        "robot_mass_kg": np.asarray(mass),
        "friction_coefficient": np.asarray(args.friction),
        "foot_half_length_m": np.asarray(config.foot_half_length_m),
        "foot_half_width_m": np.asarray(config.foot_half_width_m),
        "swing_clearance_m": np.asarray(config.swing_clearance_m),
        "model_version": np.asarray("closed_chain_ds_ss_ds_v3"),
        "accepted": np.asarray(bool(plan.metrics["accepted"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **payload)

    report = {
        "model": "closed_chain_ds_ss_ds_v3",
        "source_trajectory": str(args.input),
        "output_reference": str(args.output),
        "trigger_index": trigger,
        "trigger_angle_deg": (
            None if trigger is None else float(np.rad2deg(angles[trigger]))
        ),
        "candidate": {
            "swing_foot": plan.candidate.swing_foot,
            "lift_index": plan.candidate.lift_index,
            "land_index": plan.candidate.land_index,
            "landing_position_w": plan.candidate.landing_position_w.tolist(),
            "landing_yaw_deg": float(np.rad2deg(plan.candidate.landing_yaw_rad)),
        },
        "metrics": plan.metrics,
        "sonic_interface": {
            "tracked_references": [
                "right_wrist_pose",
                "pelvis_pose",
                "center_of_mass",
                "left_foot_pose",
                "right_foot_pose",
                "support_mode",
            ],
            "note": (
                "SONIC is treated as the downstream whole-body motion tracker; "
                "this file plans the reference and contact schedule."
            ),
        },
    }
    args.report.write_text(json.dumps(report, indent=2))

    print(
        f"accepted={plan.metrics['accepted']} swing={plan.candidate.swing_foot} "
        f"lift={plan.candidate.lift_index} land={plan.candidate.land_index}"
    )
    print(
        "arm reach improvement="
        f"{1000 * float(plan.metrics['arm_reach_improvement_m']):.1f} mm"
    )
    print(
        "min support margin="
        f"{1000 * float(plan.metrics['minimum_support_margin_m']):.1f} mm"
    )
    print(f"Wrote {args.output} and {args.report}")
    if args.fail_on_infeasible and not bool(plan.metrics["accepted"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
