#!/usr/bin/env python3
"""Independent physics audit for a G1 refrigerator trajectory.

The audit reconstructs door reaction force, solves bilateral sole-corner
contact forces, checks friction/COP and right-arm conditioning, and reports the
first angle where fixed support should be replaced by an explicit step.
"""

from __future__ import annotations
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from math_model.horizon import joint_limit_margin, manipulability_metrics
from math_model.quasistatic import DoorResistanceModel, FootPatch, fit_hinge_circle, required_tangential_hand_force, solve_contact_equilibrium
from generate_action_keyframes import BODY_JOINTS, ROOT as PROJECT_ROOT, URDF_PATH, UrdfKinematics

DEFAULT_INPUT = PROJECT_ROOT / "outputs/isaac_full_horizon_fixed_support_action.npz"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/full_horizon_quasistatic_audit.json"
FOOT_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
ARM_NAMES = (
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


def pose_from_wxyz(v: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = v[:3]
    pose[:3, :3] = Rotation.from_quat(v[[4, 5, 6, 3]]).as_matrix()
    return pose


def palm_jacobian(model, base, q, names, epsilon=2e-6):
    palm0 = model.forward(base, q)["right_palm_link"]
    jacobian = np.zeros((6, len(names)))
    for column, name in enumerate(names):
        perturbed = dict(q)
        perturbed[name] += epsilon
        palm = model.forward(base, perturbed)["right_palm_link"]
        jacobian[:3, column] = (palm[:3, 3] - palm0[:3, 3]) / epsilon
        jacobian[3:, column] = Rotation.from_matrix(palm0[:3, :3].T @ palm[:3, :3]).as_rotvec() / epsilon
    return jacobian


def parse_effort_limits(path: Path) -> dict[str, float]:
    root = ET.parse(path).getroot()
    output = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is None:
            continue
        effort = float(limit.attrib.get("effort", "nan"))
        if np.isfinite(effort) and effort > 0:
            output[joint.attrib["name"]] = effort
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--friction", type=float, default=0.7)
    parser.add_argument("--breakaway-torque", type=float, default=6.0)
    parser.add_argument("--coulomb-torque", type=float, default=3.0)
    parser.add_argument("--spring-torque-per-rad", type=float, default=0.5)
    parser.add_argument("--sigma-min-limit", type=float, default=0.012)
    parser.add_argument("--joint-margin-limit", type=float, default=0.06)
    parser.add_argument("--support-margin-limit", type=float, default=0.01)
    parser.add_argument("--fail-on-step-required", action="store_true")
    args = parser.parse_args()

    data = np.load(args.input)
    required = {"door_angle_rad", "handle_positions_w", "robot_root_pose_wxyz", "robot_joint_names", "robot_joint_positions"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise KeyError(f"trajectory is missing: {missing}")

    angles = np.asarray(data["door_angle_rad"])
    handles = np.asarray(data["handle_positions_w"])
    names = [str(name) for name in data["robot_joint_names"]]
    model = UrdfKinematics(URDF_PATH)
    joint_lookup = {joint.name: joint for joint in model.joints}
    effort_limits = parse_effort_limits(URDF_PATH)
    mass = float(sum(value[0] for value in model.inertials.values()))
    hinge = fit_hinge_circle(handles)
    door = DoorResistanceModel(args.breakaway_torque, args.coulomb_torque, args.spring_torque_per_rad)
    lower = np.asarray([joint_lookup[name].lower for name in BODY_JOINTS])
    upper = np.asarray([joint_lookup[name].upper for name in BODY_JOINTS])
    initial_feet = None
    records = []

    for frame, (angle, handle, root_values, q_values) in enumerate(zip(angles, handles, data["robot_root_pose_wxyz"], data["robot_joint_positions"])):
        base = pose_from_wxyz(root_values)
        q = dict(zip(names, q_values))
        poses = model.forward(base, q)
        com = model.center_of_mass(poses)
        if initial_feet is None:
            initial_feet = {name: poses[name].copy() for name in FOOT_NAMES}

        feet, drift = [], []
        for name, label in zip(FOOT_NAMES, ("left", "right")):
            foot = poses[name]
            center = foot[:3, 3] + foot[:3, :3] @ np.array([0.0, 0.0, -0.035])
            feet.append(FootPatch(center, foot[:3, :3], 0.12, 0.05, label))
            initial = initial_feet[name]
            drift.append(np.linalg.norm(foot[:3, 3] - initial[:3, 3]))

        door_torque = door.torque(float(angle))
        hand_force = required_tangential_hand_force(hinge.center, hinge.axis, handle, door_torque)
        equilibrium = solve_contact_equilibrium(
            mass=mass,
            center_of_mass=com,
            hand_point=handle,
            hand_force_on_robot=hand_force,
            feet=feet,
            friction_coefficient=args.friction,
        )
        arm = manipulability_metrics(palm_jacobian(model, base, q, ARM_NAMES))
        q_body = np.asarray([q[name] for name in BODY_JOINTS])
        minimum_margin = float(np.min(joint_limit_margin(q_body, lower, upper)))

        shoulder = poses["right_shoulder_roll_link"][:3, 3]
        elbow = poses["right_elbow_link"][:3, 3]
        wrist = poses["right_wrist_yaw_link"][:3, 3]
        palm = poses["right_palm_link"][:3, 3]
        a, b = shoulder - elbow, wrist - elbow
        elbow_angle = float(np.rad2deg(np.arccos(np.clip(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9), -1.0, 1.0))))
        reach = float(np.linalg.norm(shoulder - palm))

        reasons = []
        if not equilibrium.feasible:
            reasons.append("static_contact_equilibrium")
        if equilibrium.support_margin < args.support_margin_limit:
            reasons.append("center_of_pressure_margin")
        if arm.sigma_min < args.sigma_min_limit:
            reasons.append("arm_jacobian_singularity")
        if minimum_margin < args.joint_margin_limit:
            reasons.append("joint_limit_margin")
        if max(drift) > 0.002:
            reasons.append("support_foot_drift")
        if elbow_angle < 75.0 or elbow_angle > 170.0 or reach > 0.40:
            reasons.append("arm_geometry_margin")

        records.append({
            "frame": frame,
            "door_angle_deg": float(np.rad2deg(angle)),
            "door_torque_nm": door_torque,
            "hand_reaction_force_n": hand_force.tolist(),
            "static_equilibrium_feasible": equilibrium.feasible,
            "minimum_friction_margin_n": equilibrium.minimum_friction_margin,
            "center_of_pressure_w": equilibrium.center_of_pressure.tolist(),
            "support_margin_m": equilibrium.support_margin,
            "arm_sigma_min": arm.sigma_min,
            "arm_condition_number": arm.condition_number,
            "minimum_normalized_joint_margin": minimum_margin,
            "elbow_bend_deg": elbow_angle,
            "shoulder_to_palm_m": reach,
            "maximum_support_foot_drift_m": float(max(drift)),
            "requires_contact_mode_change": bool(reasons),
            "failure_reasons": reasons,
        })

    failing = [record for record in records if record["requires_contact_mode_change"]]
    first = failing[0] if failing else None
    report = {
        "model": "full_horizon_quasistatic_v2",
        "trajectory": str(args.input),
        "robot_mass_kg": mass,
        "hinge_fit": {"center_w": hinge.center.tolist(), "axis_w": hinge.axis.tolist(), "radius_m": hinge.radius, "rms_error_m": hinge.rms_error},
        "assumptions": {"friction_coefficient": args.friction, "fixed_double_support": True, "quasi_static": True},
        "fixed_support_valid_for_full_horizon": first is None,
        "first_required_contact_mode_change_deg": None if first is None else first["door_angle_deg"],
        "first_failure_reasons": [] if first is None else first["failure_reasons"],
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    if first is None:
        print("Fixed double support passed the complete quasi-static audit.")
    else:
        print(f"Contact-mode change required at {first['door_angle_deg']:.2f} deg: {', '.join(first['failure_reasons'])}")
    print(f"Wrote {args.output}")
    if args.fail_on_step_required and first is not None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
