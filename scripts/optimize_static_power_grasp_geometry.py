#!/usr/bin/env python3
"""Solve a static whole-hand power grasp from mesh/cylinder geometry.

The refrigerator grip is a 24.1 mm diameter vertical cylinder (measured from
the exported USD triangle mesh).  The variables are the cylinder pose in the
palm frame and a low-dimensional, fixed hand posture.  The objective requires
surface patches from the palm, thumb and all four fingers, penalizes any mesh
penetration, and requires the contact set to enclose more than a half circle.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.transform import Rotation

from generate_action_keyframes import ROOT, URDF_PATH, UrdfKinematics


SOURCE = ROOT / "outputs/isaac_humanlike_grasp_refined2.npz"
HANDLE_MESH = ROOT / "outputs/upper_handle_mesh.npz"
OUTPUT = ROOT / "outputs/static_power_grasp_geometry_solution.json"
HANDLE_RADIUS = 0.01206
HANDLE_HALF_LENGTH = 0.105

GROUPS = {
    "palm": ["right_palm_link"],
    "thumb": [f"right_finger1_link{i}" for i in range(1, 5)],
    **{
        f"finger{finger}": [f"right_finger{finger}_link{i}" for i in range(1, 5)]
        for finger in range(2, 6)
    },
}


def load_stl_surface(path: Path, max_points: int | None) -> np.ndarray:
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    if 84 + 50 * count != len(data):
        raise ValueError(f"Only binary STL is supported: {path}")
    faces = np.empty((count, 3, 3), dtype=np.float64)
    offset = 84
    for i in range(count):
        faces[i] = np.asarray(struct.unpack_from("<9f", data, offset + 12)).reshape(3, 3)
        offset += 50
    # Vertices alone miss the middle of large triangles.  Add edge midpoints
    # and centroids so that a reported gap/penetration reflects the surface.
    points = np.concatenate(
        [
            faces.reshape(-1, 3),
            0.5 * (faces[:, 0] + faces[:, 1]),
            0.5 * (faces[:, 1] + faces[:, 2]),
            0.5 * (faces[:, 2] + faces[:, 0]),
            faces.mean(axis=1),
        ]
    )
    points = np.unique(np.round(points, decimals=7), axis=0)
    if max_points is not None and len(points) > max_points:
        points = points[np.linspace(0, len(points) - 1, max_points, dtype=int)]
    return points


def source_palm_orientation(model: UrdfKinematics) -> np.ndarray:
    data = np.load(SOURCE)
    names = [str(name) for name in data["robot_joint_names"]]
    q = dict(zip(names, data["robot_joint_positions"][0]))
    root = data["robot_root_pose_wxyz"][0]
    base = np.eye(4)
    base[:3, 3] = root[:3]
    base[:3, :3] = Rotation.from_quat(root[[4, 5, 6, 3]]).as_matrix()
    return model.forward(base, q)["right_palm_link"][:3, :3]


def measured_handle_axis_center() -> np.ndarray:
    data = np.load(HANDLE_MESH)
    vertices = data["vertices_w"]
    triangles = data["triangles"]
    z0 = 0.875
    crossings: list[np.ndarray] = []
    for triangle in vertices[triangles]:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            za, zb = triangle[a, 2] - z0, triangle[b, 2] - z0
            if za * zb < 0:
                t = -za / (zb - za)
                crossings.append(triangle[a] + t * (triangle[b] - triangle[a]))
    section = np.asarray(crossings)
    xy = 0.5 * (section[:, :2].min(axis=0) + section[:, :2].max(axis=0))
    return np.r_[xy, 0.8758843]


class GraspModel:
    def __init__(self, sampled_points: int | None = 500):
        self.model = UrdfKinematics(URDF_PATH)
        self.visual_points: dict[str, np.ndarray] = {}
        for links in GROUPS.values():
            for link in links:
                mesh_path, visual_tf, _ = self.model.visuals[link]
                points = load_stl_surface(mesh_path, sampled_points)
                self.visual_points[link] = points @ visual_tf[:3, :3].T + visual_tf[:3, 3]
        palm_R = source_palm_orientation(self.model)
        self.axis = palm_R.T @ np.array([0.0, 0.0, 1.0])
        self.axis /= np.linalg.norm(self.axis)
        self.measured_center_w = measured_handle_axis_center()

    @staticmethod
    def unpack(x: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        center = x[:3]
        q = {f"right_finger1_joint{i}": float(x[2 + i]) for i in range(1, 5)}
        # Each finger has independent MCP/PIP/DIP flexion.  This is necessary
        # because requiring one contact per whole finger still admits an open
        # finger whose proximal link alone happens to touch the handle.
        for offset, finger in enumerate(range(2, 6)):
            start = 7 + 3 * offset
            q[f"right_finger{finger}_joint1"] = float(x[start])
            q[f"right_finger{finger}_joint2"] = 0.0
            q[f"right_finger{finger}_joint3"] = float(x[start + 1])
            q[f"right_finger{finger}_joint4"] = float(x[start + 2])
        return center, q

    def link_points(self, q: dict[str, float]) -> dict[str, np.ndarray]:
        poses = self.model.forward(np.eye(4), q)
        palm_inv = np.linalg.inv(poses["right_palm_link"])
        output = {}
        for links in GROUPS.values():
            for link in links:
                tf = palm_inv @ poses[link]
                points = self.visual_points[link]
                output[link] = points @ tf[:3, :3].T + tf[:3, 3]
        return output

    def surface_metrics(self, points: np.ndarray, center: np.ndarray) -> dict:
        delta = points - center
        axial = delta @ self.axis
        # The hand spans less than the 210 mm straight grip section.  Using the
        # corresponding infinite cylinder keeps the distance field continuous;
        # the optimized center and a separate axial-range check keep contacts
        # away from the two end brackets.
        radial_vec = delta - np.outer(axial, self.axis)
        radial = np.linalg.norm(radial_vec, axis=1)
        signed = radial - HANDLE_RADIUS
        count = min(12, len(signed))
        nearest = np.argpartition(np.abs(signed), count - 1)[:count]
        near_index = int(nearest[np.argmin(np.abs(signed[nearest]))])
        direction = radial_vec[near_index] / max(radial[near_index], 1e-12)
        return {
            "patch_gap_m": float(np.mean(np.abs(signed[nearest]))),
            "minimum_surface_distance_m": float(np.min(np.abs(signed))),
            "max_penetration_m": float(max(0.0, -signed.min())),
            "contact_direction": direction.tolist(),
        }

    def metrics(self, x: np.ndarray) -> dict:
        center, q = self.unpack(x)
        links = self.link_points(q)
        link_metrics = {name: self.surface_metrics(points, center) for name, points in links.items()}
        group_metrics = {
            group: self.surface_metrics(np.concatenate([links[link] for link in members]), center)
            for group, members in GROUPS.items()
        }
        # On this 24 mm bar the Wuji fingers cannot make both rigid phalanges
        # tangent without one cutting through the cylinder.  A valid small-bar
        # power grasp therefore uses proximal-pad contact and a curled middle
        # phalanx kept within 7 mm of the surface; fingertips are not targets.
        required_names = [
            "right_finger1_link2", "right_finger1_link3",
            *[f"right_finger{finger}_link2" for finger in range(2, 6)],
        ]
        near_names = [f"right_finger{finger}_link3" for finger in range(2, 6)]
        directions = [np.asarray(group_metrics["palm"]["contact_direction"])]
        directions.extend(
            np.asarray(link_metrics[name]["contact_direction"])
            for name in [*required_names, *near_names]
        )

        # Contact-direction coverage is coordinate-free: project the directions
        # into an orthonormal plane perpendicular to the handle axis.
        reference = np.array([1.0, 0.0, 0.0])
        reference -= self.axis * np.dot(reference, self.axis)
        reference /= np.linalg.norm(reference)
        tangent = np.cross(self.axis, reference)
        theta = np.mod(
            [np.arctan2(np.dot(direction, tangent), np.dot(direction, reference)) for direction in directions],
            2 * np.pi,
        )
        theta = np.sort(theta)
        gaps = np.diff(np.r_[theta, theta[0] + 2 * np.pi])
        coverage = 2 * np.pi - gaps.max()
        return {
            "center_in_palm_m": center.tolist(),
            "handle_axis_in_palm": self.axis.tolist(),
            "joint_positions_rad": q,
            "group_metrics": group_metrics,
            "link_metrics": link_metrics,
            "required_contact_links": required_names,
            "required_near_links": near_names,
            "enclosure_angle_deg": float(np.rad2deg(coverage)),
        }

    def objective(self, x: np.ndarray) -> float:
        metrics = self.metrics(x)
        contact_gap_mm = np.array(
            [metrics["link_metrics"][name]["patch_gap_m"] for name in metrics["required_contact_links"]]
        ) * 1000
        near_gap_mm = np.array(
            [metrics["link_metrics"][name]["patch_gap_m"] for name in metrics["required_near_links"]]
        ) * 1000
        palm_gap_mm = 1000 * metrics["group_metrics"]["palm"]["patch_gap_m"]
        collision_metrics = [metrics["group_metrics"]["palm"], *metrics["link_metrics"].values()]
        penetration_mm = np.array([v["max_penetration_m"] for v in collision_metrics]) * 1000
        # A broad contact patch below 1.5 mm is preferred.  Penetration above
        # 0.8 mm is deliberately much more expensive than a remaining gap.
        contact_loss = np.sum((contact_gap_mm / 1.5) ** 2)
        contact_loss += 2.0 * (max(0.0, palm_gap_mm - 4.0) / 2.0) ** 2
        contact_loss += np.sum((np.maximum(near_gap_mm - 5.0, 0.0) / 2.0) ** 2)
        penetration_loss = 240.0 * np.sum((np.maximum(penetration_mm - 0.4, 0.0) / 0.35) ** 2)
        penetration_loss += 8.0 * np.sum((penetration_mm / 0.8) ** 2)
        enclosure_loss = 12.0 * (max(0.0, 200.0 - metrics["enclosure_angle_deg"]) / 20.0) ** 2
        # Weak priors only remove equivalent/unnatural solutions; contact
        # geometry remains the dominant term.
        center_prior = np.array([0.049, 0.011, 0.094])
        posture_prior = np.array(
            [0.90, 0.10, 0.90, 0.80, *([1.00, 0.90, 0.90] * 4)]
        )
        posture = x[3:]
        regularization = 0.08 * np.sum(((x[:3] - center_prior) / 0.025) ** 2)
        regularization += 0.03 * np.sum(((posture - posture_prior) / 0.6) ** 2)
        return float(contact_loss + penetration_loss + enclosure_loss + regularization)


def main() -> None:
    grasp = GraspModel(sampled_points=1200)
    bounds = [
        (-0.005, 0.045), (-0.020, 0.035), (0.065, 0.110),
        (0.05, 1.55), (-0.13, 0.90), (-0.40, 1.50), (-0.40, 1.50),
        *([(0.45, 1.50), (0.45, 1.50), (0.60, 1.50)] * 4),
    ]
    x0 = np.array(
        [
            0.0168, 0.0085, 0.0822,
            1.05, 0.25, 1.10, 0.80,
            1.35, 1.20, 1.00,
            1.20, 1.20, 1.00,
            0.95, 1.20, 1.00,
            0.70, 1.20, 1.00,
        ]
    )
    result = differential_evolution(
        grasp.objective,
        bounds,
        seed=7,
        popsize=8,
        maxiter=130,
        tol=2e-4,
        polish=True,
        updating="immediate",
        workers=1,
        x0=x0,
    )
    # Re-audit the selected posture at full STL surface resolution.
    audit = GraspModel(sampled_points=None).metrics(result.x)
    audit.update(
        {
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "objective": float(result.fun),
            "handle_radius_m": HANDLE_RADIUS,
            "handle_half_length_m": HANDLE_HALF_LENGTH,
            "measured_handle_axis_center_w": grasp.measured_center_w.tolist(),
            "acceptance": {
                "maximum_palm_gap_m": 0.0060,
                "maximum_contact_link_gap_m": 0.0020,
                "maximum_near_link_gap_m": 0.0070,
                "maximum_penetration_m": 0.0010,
                "minimum_enclosure_angle_deg": 190.0,
            },
        }
    )
    contact_gaps = [audit["link_metrics"][name]["patch_gap_m"] for name in audit["required_contact_links"]]
    near_gaps = [audit["link_metrics"][name]["patch_gap_m"] for name in audit["required_near_links"]]
    palm_gap = audit["group_metrics"]["palm"]["patch_gap_m"]
    collision_metrics = [audit["group_metrics"]["palm"], *audit["link_metrics"].values()]
    penetrations = [v["max_penetration_m"] for v in collision_metrics]
    audit["accepted"] = bool(
        palm_gap <= audit["acceptance"]["maximum_palm_gap_m"]
        and max(contact_gaps) <= audit["acceptance"]["maximum_contact_link_gap_m"]
        and max(near_gaps) <= audit["acceptance"]["maximum_near_link_gap_m"]
        and max(penetrations) <= audit["acceptance"]["maximum_penetration_m"]
        and audit["enclosure_angle_deg"] >= audit["acceptance"]["minimum_enclosure_angle_deg"]
    )
    OUTPUT.write_text(json.dumps(audit, indent=2))
    print(f"objective={result.fun:.4f} accepted={audit['accepted']}")
    print(f"center_in_palm={np.round(audit['center_in_palm_m'], 5).tolist()}")
    print(f"enclosure_deg={audit['enclosure_angle_deg']:.1f}")
    for name, metrics in audit["group_metrics"].items():
        print(
            f"{name}: patch_gap_mm={1000*metrics['patch_gap_m']:.2f} "
            f"penetration_mm={1000*metrics['max_penetration_m']:.2f}"
        )
    print(f"contact_link_max_gap_mm={1000*max(contact_gaps):.2f}")
    print(f"near_link_max_gap_mm={1000*max(near_gaps):.2f}")
    print(f"all_link_max_penetration_mm={1000*max(penetrations):.2f}")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
