#!/usr/bin/env python3
"""Re-audit the final whole-body trajectory against the full hand meshes."""

from __future__ import annotations

import json

import numpy as np

from generate_action_keyframes import ROOT
from optimize_static_power_grasp_geometry import GraspModel


TRAJECTORY = ROOT / "outputs/isaac_static_power_grasp.npz"
OUTPUT = ROOT / "outputs/static_power_grasp_final_validation.json"


def main() -> None:
    trajectory = np.load(TRAJECTORY)
    names = [str(name) for name in trajectory["robot_joint_names"]]
    q = dict(zip(names, trajectory["robot_joint_positions"][0]))
    x = np.r_[
        trajectory["handle_in_palm_actual"],
        [q[f"right_finger1_joint{i}"] for i in range(1, 5)],
        [
            q[f"right_finger{finger}_joint{joint}"]
            for finger in range(2, 6)
            for joint in (1, 3, 4)
        ],
    ]
    report = GraspModel(sampled_points=None).metrics(x)
    report["acceptance"] = {
        "maximum_palm_gap_m": 0.0060,
        "maximum_contact_link_gap_m": 0.0020,
        "maximum_near_link_gap_m": 0.0070,
        "maximum_penetration_m": 0.0010,
        "minimum_enclosure_angle_deg": 190.0,
        "maximum_palm_target_error_m": 0.0005,
    }
    contact_gaps = [
        report["link_metrics"][name]["patch_gap_m"] for name in report["required_contact_links"]
    ]
    near_gaps = [
        report["link_metrics"][name]["patch_gap_m"] for name in report["required_near_links"]
    ]
    palm_gap = report["group_metrics"]["palm"]["patch_gap_m"]
    collision_metrics = [report["group_metrics"]["palm"], *report["link_metrics"].values()]
    penetrations = [metrics["max_penetration_m"] for metrics in collision_metrics]
    palm_error = float(
        np.linalg.norm(trajectory["handle_in_palm_actual"] - trajectory["handle_in_palm_target"])
    )
    report["palm_target_error_m"] = palm_error
    report["body_posture_deg"] = {
        "waist_yaw": float(np.rad2deg(q["waist_yaw_joint"])),
        "waist_roll": float(np.rad2deg(q["waist_roll_joint"])),
        "waist_pitch": float(np.rad2deg(q["waist_pitch_joint"])),
        "left_knee": float(np.rad2deg(q["left_knee_joint"])),
        "right_knee": float(np.rad2deg(q["right_knee_joint"])),
        "right_wrist_roll": float(np.rad2deg(q["right_wrist_roll_joint"])),
        "right_wrist_pitch": float(np.rad2deg(q["right_wrist_pitch_joint"])),
        "right_wrist_yaw": float(np.rad2deg(q["right_wrist_yaw_joint"])),
    }
    report["accepted"] = bool(
        palm_gap <= report["acceptance"]["maximum_palm_gap_m"]
        and max(contact_gaps) <= report["acceptance"]["maximum_contact_link_gap_m"]
        and max(near_gaps) <= report["acceptance"]["maximum_near_link_gap_m"]
        and max(penetrations) <= report["acceptance"]["maximum_penetration_m"]
        and report["enclosure_angle_deg"] >= report["acceptance"]["minimum_enclosure_angle_deg"]
        and palm_error <= report["acceptance"]["maximum_palm_target_error_m"]
    )
    OUTPUT.write_text(json.dumps(report, indent=2))
    print(f"accepted={report['accepted']} palm_target_error_mm={1000*palm_error:.2f}")
    print(f"enclosure_deg={report['enclosure_angle_deg']:.1f}")
    for name, metrics in report["group_metrics"].items():
        print(
            f"{name}: patch_gap_mm={1000*metrics['patch_gap_m']:.2f} "
            f"penetration_mm={1000*metrics['max_penetration_m']:.2f}"
        )
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
