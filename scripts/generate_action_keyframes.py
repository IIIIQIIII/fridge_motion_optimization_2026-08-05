#!/usr/bin/env python3
"""Generate optimized G1 fridge-opening keyframes directly from the URDF.

This is a kinematic prototype. It does not use a learned policy or motion clip.
"""

from __future__ import annotations

import csv
import math
import struct
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
COORDEX_ROOT = ROOT.parent
ASSET_ROOT = COORDEX_ROOT / "source_repo/source/coordex/coordex/assets/g1_wuji"
URDF_PATH = ASSET_ROOT / "g1_wuji.urdf"
OUTPUT_DIR = ROOT / "outputs"


def vec(text: str | None, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=float)
    return np.asarray([float(x) for x in text.split()], dtype=float)


def transform(xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    out[:3, 3] = xyz
    return out


def axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
    return out


def pose_error(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    position = actual[:3, 3] - target[:3, 3]
    orientation = Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).as_rotvec()
    return np.r_[position, orientation]


@dataclass
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


class UrdfKinematics:
    def __init__(self, path: Path):
        root = ET.parse(path).getroot()
        self.path = path
        self.joints: list[Joint] = []
        self.children: dict[str, list[Joint]] = {}
        self.visuals: dict[str, tuple[Path, np.ndarray, str]] = {}
        self.inertials: dict[str, tuple[float, np.ndarray]] = {}

        for link in root.findall("link"):
            name = link.attrib["name"]
            inertial = link.find("inertial")
            if inertial is not None and inertial.find("mass") is not None:
                mass = float(inertial.find("mass").attrib["value"])
                origin = inertial.find("origin")
                com = vec(origin.attrib.get("xyz") if origin is not None else None)
                self.inertials[name] = (mass, com)
            visual = link.find("visual")
            if visual is not None:
                mesh = visual.find("geometry/mesh")
                if mesh is not None:
                    origin = visual.find("origin")
                    visual_tf = transform(
                        vec(origin.attrib.get("xyz") if origin is not None else None),
                        vec(origin.attrib.get("rpy") if origin is not None else None),
                    )
                    material = visual.find("material")
                    material_name = material.attrib.get("name", "white") if material is not None else "white"
                    self.visuals[name] = (path.parent / mesh.attrib["filename"], visual_tf, material_name)

        for node in root.findall("joint"):
            origin = node.find("origin")
            axis_node = node.find("axis")
            axis = vec(axis_node.attrib.get("xyz") if axis_node is not None else None, (1, 0, 0))
            axis = axis / max(np.linalg.norm(axis), 1e-12)
            limit = node.find("limit")
            kind = node.attrib["type"]
            lower, upper = -math.pi, math.pi
            if limit is not None and kind != "fixed":
                lower = float(limit.attrib.get("lower", -math.pi))
                upper = float(limit.attrib.get("upper", math.pi))
            joint = Joint(
                name=node.attrib["name"],
                kind=kind,
                parent=node.find("parent").attrib["link"],
                child=node.find("child").attrib["link"],
                origin=transform(
                    vec(origin.attrib.get("xyz") if origin is not None else None),
                    vec(origin.attrib.get("rpy") if origin is not None else None),
                ),
                axis=axis,
                lower=lower,
                upper=upper,
            )
            self.joints.append(joint)
            self.children.setdefault(joint.parent, []).append(joint)

    def forward(self, base_tf: np.ndarray, q: dict[str, float]) -> dict[str, np.ndarray]:
        poses = {"pelvis": base_tf}
        stack = ["pelvis"]
        while stack:
            parent = stack.pop()
            for joint in self.children.get(parent, []):
                joint_tf = joint.origin
                if joint.kind in {"revolute", "continuous"}:
                    joint_tf = joint_tf @ axis_rotation(joint.axis, q.get(joint.name, 0.0))
                poses[joint.child] = poses[parent] @ joint_tf
                stack.append(joint.child)
        return poses

    def center_of_mass(self, poses: dict[str, np.ndarray]) -> np.ndarray:
        weighted = np.zeros(3)
        total = 0.0
        for link, (mass, local_com) in self.inertials.items():
            if link not in poses:
                continue
            p = poses[link][:3, :3] @ local_com + poses[link][:3, 3]
            weighted += mass * p
            total += mass
        return weighted / total


BODY_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def initial_joint_map() -> dict[str, float]:
    q = {name: 0.0 for name in BODY_JOINTS}
    for side in ("left", "right"):
        q[f"{side}_hip_pitch_joint"] = -0.312
        q[f"{side}_knee_joint"] = 0.669
        q[f"{side}_ankle_pitch_joint"] = -0.363
        q[f"{side}_elbow_joint"] = 0.6
        q[f"{side}_shoulder_pitch_joint"] = 0.2
    q["left_shoulder_roll_joint"] = 0.2
    q["right_shoulder_roll_joint"] = -0.2
    return q


def unpack(x: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    base = transform(x[:3], x[3:6])
    return base, dict(zip(BODY_JOINTS, x[6:]))


def make_initial_vector(q0: dict[str, float]) -> np.ndarray:
    return np.r_[np.array([0.0, 0.0, 0.76, 0.0, 0.0, 0.0]), [q0[name] for name in BODY_JOINTS]]


def door_handle(hinge: np.ndarray, radius: float, angle: float, height: float) -> np.ndarray:
    radial0 = np.array([-radius, 0.0, 0.0])
    radial = Rotation.from_euler("z", angle).apply(radial0)
    return hinge + radial + np.array([0.0, 0.0, height])


def solve_keyframes(model: UrdfKinematics):
    q0 = initial_joint_map()
    x0 = make_initial_vector(q0)
    base0, qmap0 = unpack(x0)
    poses0 = model.forward(base0, qmap0)
    foot_targets = {
        name: poses0[name].copy()
        for name in ("left_ankle_roll_link", "right_ankle_roll_link")
    }

    # A reachable first-pass appliance placement. These values define the
    # prototype scene and are not extracted from the refrigerator USD yet.
    hinge = np.array([0.85, -0.25, 0.0])
    radius = 0.32
    handle_height = 1.02
    angles = np.deg2rad([0, 15, 30, 45, 60])

    joint_lookup = {j.name: j for j in model.joints}
    lower = np.r_[[-0.30, -0.26, 0.64, -0.30, -0.30, -0.40], [joint_lookup[n].lower for n in BODY_JOINTS]]
    upper = np.r_[[0.30, 0.26, 0.88, 0.30, 0.30, 0.40], [joint_lookup[n].upper for n in BODY_JOINTS]]

    posture_weights = np.ones(len(BODY_JOINTS)) * 0.45
    for i, name in enumerate(BODY_JOINTS):
        if name.startswith("right_") and any(k in name for k in ("shoulder", "elbow", "wrist")):
            posture_weights[i] = 0.12
        elif name.startswith("left_") and any(k in name for k in ("shoulder", "elbow", "wrist")):
            posture_weights[i] = 0.8
        elif name.startswith("waist_"):
            posture_weights[i] = 0.18

    results = []
    previous = x0.copy()
    support_center = 0.5 * (
        foot_targets["left_ankle_roll_link"][:2, 3]
        + foot_targets["right_ankle_roll_link"][:2, 3]
    )

    for frame_idx, angle in enumerate(angles):
        target = door_handle(hinge, radius, angle, handle_height)

        def residual(x):
            base, qmap = unpack(x)
            poses = model.forward(base, qmap)
            palm = poses["right_palm_link"]
            com = model.center_of_mass(poses)
            parts = [40.0 * (palm[:3, 3] - target)]
            for foot_name, foot_target in foot_targets.items():
                err = pose_error(poses[foot_name], foot_target)
                parts.append(np.r_[70.0 * err[:3], 28.0 * err[3:]])
            parts.append(2.0 * (com[:2] - support_center))
            parts.append(np.array([1.1, 1.1, 1.4, 0.8, 0.8, 0.8]) * (x[:6] - x0[:6]))
            parts.append(posture_weights * (x[6:] - x0[6:]))
            if frame_idx > 0:
                continuity_weights = np.r_[np.ones(6) * 0.9, np.ones(len(BODY_JOINTS)) * 0.22]
                parts.append(continuity_weights * (x - previous))
            return np.concatenate(parts)

        solution = least_squares(
            residual,
            previous,
            bounds=(lower, upper),
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
            max_nfev=1400,
            verbose=0,
        )
        base, qmap = unpack(solution.x)
        poses = model.forward(base, qmap)
        palm_error = float(np.linalg.norm(poses["right_palm_link"][:3, 3] - target))
        foot_error = max(
            float(np.linalg.norm(poses[name][:3, 3] - foot_targets[name][:3, 3]))
            for name in foot_targets
        )
        results.append({
            "angle": angle,
            "target": target,
            "x": solution.x.copy(),
            "poses": poses,
            "palm_error": palm_error,
            "foot_error": foot_error,
            "cost": float(solution.cost),
            "success": bool(solution.success),
        })
        previous = solution.x.copy()
    return results, hinge, radius, handle_height


def load_binary_stl(path: Path, max_faces=180) -> np.ndarray | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) < 84:
        return None
    count = struct.unpack_from("<I", data, 80)[0]
    if 84 + 50 * count != len(data):
        return None
    faces = np.empty((count, 3, 3), dtype=np.float32)
    offset = 84
    for i in range(count):
        faces[i] = np.asarray(struct.unpack_from("<9f", data, offset + 12)).reshape(3, 3)
        offset += 50
    if count > max_faces:
        indices = np.linspace(0, count - 1, max_faces, dtype=int)
        faces = faces[indices]
    return faces


def transformed_faces(faces: np.ndarray, tf: np.ndarray) -> np.ndarray:
    return faces @ tf[:3, :3].T + tf[:3, 3]


def draw_fridge(ax, hinge, radius, angle, handle_height, alpha=1.0):
    cabinet_x = [0.94, 1.28]
    cabinet_y = [-0.35, 0.35]
    cabinet_z = [0.0, 1.75]
    for z in cabinet_z:
        ax.plot([cabinet_x[0], cabinet_x[1], cabinet_x[1], cabinet_x[0], cabinet_x[0]],
                [cabinet_y[0], cabinet_y[0], cabinet_y[1], cabinet_y[1], cabinet_y[0]],
                [z] * 5, color="#707780", lw=1, alpha=0.55 * alpha)
    for x in cabinet_x:
        for y in cabinet_y:
            ax.plot([x, x], [y, y], cabinet_z, color="#707780", lw=1, alpha=0.55 * alpha)

    radial = Rotation.from_euler("z", angle).apply(np.array([-radius, 0.0, 0.0]))
    outer = hinge + radial
    panel = np.array([
        [hinge[0], hinge[1], 0.1], [outer[0], outer[1], 0.1],
        [outer[0], outer[1], 1.65], [hinge[0], hinge[1], 1.65],
    ])
    ax.add_collection3d(Poly3DCollection([panel], facecolor="#4c86c6", edgecolor="#24517d", alpha=0.10 * alpha))
    ax.plot(panel[[0, 1, 2, 3, 0], 0], panel[[0, 1, 2, 3, 0], 1], panel[[0, 1, 2, 3, 0], 2],
            color="#24517d", lw=2, alpha=alpha)
    handle = door_handle(hinge, radius, angle, handle_height)
    ax.scatter(*handle, s=35, color="#d84a3a", depthshade=False, zorder=10)


def draw_robot(ax, model: UrdfKinematics, poses, alpha=1.0, mesh_cache=None):
    mesh_cache = mesh_cache if mesh_cache is not None else {}
    skip_tokens = ("finger", "contour", "constraint", "sensor", "camera", "imu")
    for link, (mesh_path, visual_tf, material) in model.visuals.items():
        if link not in poses or any(token in link for token in skip_tokens):
            continue
        if link not in mesh_cache:
            mesh_cache[link] = load_binary_stl(mesh_path)
        faces = mesh_cache[link]
        if faces is None:
            continue
        world_faces = transformed_faces(faces, poses[link] @ visual_tf)
        color = "#7f8c99" if material == "white" else "#20262d"
        collection = Poly3DCollection(world_faces, facecolor=color, edgecolor="none", alpha=alpha)
        ax.add_collection3d(collection)

    # Explicit kinematic chains make the optimized posture readable even when
    # a thin or white STL surface is hard to see at keyframe scale.
    chains = [
        ["pelvis", "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
         "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link"],
        ["pelvis", "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
         "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link"],
        ["pelvis", "waist_yaw_link", "waist_roll_link", "torso_link"],
        ["torso_link", "left_shoulder_pitch_link", "left_shoulder_roll_link",
         "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_roll_link",
         "left_wrist_pitch_link", "left_wrist_yaw_link", "left_palm_link"],
        ["torso_link", "right_shoulder_pitch_link", "right_shoulder_roll_link",
         "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_link",
         "right_wrist_pitch_link", "right_wrist_yaw_link", "right_palm_link"],
    ]
    for chain in chains:
        points = np.asarray([poses[name][:3, 3] for name in chain if name in poses])
        if len(points) >= 2:
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color="#087f6d", lw=3.0, alpha=alpha)
    palm = poses["right_palm_link"][:3, 3]
    ax.scatter(*palm, s=18, color="#16a085", depthshade=False)


def setup_axis(ax, title=None):
    ax.set_xlim(-0.28, 1.22)
    ax.set_ylim(-0.82, 0.48)
    ax.set_zlim(0.0, 1.58)
    ax.set_box_aspect((1.5, 1.3, 1.58))
    # Look from the robot side of the door so the panel does not hide the body.
    ax.view_init(elev=17, azim=-122)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=11, pad=2)


def render(results, model, hinge, radius, handle_height):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mesh_cache = {}
    fig = plt.figure(figsize=(18, 4.2), constrained_layout=True)
    for i, item in enumerate(results):
        ax = fig.add_subplot(1, len(results), i + 1, projection="3d")
        draw_fridge(ax, hinge, radius, item["angle"], handle_height)
        draw_robot(ax, model, item["poses"], mesh_cache=mesh_cache)
        setup_axis(ax, f'{math.degrees(item["angle"]):.0f}°  hand err {1000*item["palm_error"]:.1f} mm')
    fig.suptitle("Computed G1 fridge-opening keyframes (fixed feet, kinematic optimization)", fontsize=14)
    fig.savefig(OUTPUT_DIR / "g1_fridge_action_keyframes.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig = plt.figure(figsize=(7.5, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    for i, item in enumerate(results):
        opacity = 0.22 + 0.16 * i
        draw_fridge(ax, hinge, radius, item["angle"], handle_height, alpha=opacity)
        draw_robot(ax, model, item["poses"], alpha=opacity, mesh_cache=mesh_cache)
    setup_axis(ax)
    ax.set_title("Optimized motion overlay: 0° → 60°", fontsize=13)
    fig.savefig(OUTPUT_DIR / "g1_fridge_action_overlay.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    for i, item in enumerate(results):
        fig = plt.figure(figsize=(6.2, 6.2))
        ax = fig.add_subplot(111, projection="3d")
        draw_fridge(ax, hinge, radius, item["angle"], handle_height)
        draw_robot(ax, model, item["poses"], mesh_cache=mesh_cache)
        setup_axis(ax, f'G1 opening fridge — {math.degrees(item["angle"]):.0f}°')
        fig.savefig(OUTPUT_DIR / f"frame_{i:02d}.png", dpi=190, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    gif_frames = [Image.open(OUTPUT_DIR / f"frame_{i:02d}.png").convert("P", palette=Image.Palette.ADAPTIVE)
                  for i in range(len(results))]
    gif_frames[0].save(
        OUTPUT_DIR / "g1_fridge_action.gif",
        save_all=True,
        append_images=gif_frames[1:] + list(reversed(gif_frames[1:-1])),
        duration=650,
        loop=0,
        optimize=True,
    )


def save_data(results):
    xs = np.vstack([item["x"] for item in results])
    np.savez(
        OUTPUT_DIR / "g1_fridge_action.npz",
        door_angle_rad=np.asarray([item["angle"] for item in results]),
        base_xyz_rpy=xs[:, :6],
        joint_names=np.asarray(BODY_JOINTS),
        joint_positions=xs[:, 6:],
        palm_target=np.vstack([item["target"] for item in results]),
        palm_error_m=np.asarray([item["palm_error"] for item in results]),
        foot_error_m=np.asarray([item["foot_error"] for item in results]),
    )
    with (OUTPUT_DIR / "g1_fridge_action.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["door_angle_deg", "palm_error_m", "foot_error_m", "base_x", "base_y", "base_z", *BODY_JOINTS])
        for item in results:
            x = item["x"]
            writer.writerow([math.degrees(item["angle"]), item["palm_error"], item["foot_error"], *x[:3], *x[6:]])


def main():
    if not URDF_PATH.is_file():
        raise FileNotFoundError(URDF_PATH)
    model = UrdfKinematics(URDF_PATH)
    missing = [name for name in BODY_JOINTS if name not in {j.name for j in model.joints}]
    if missing:
        raise RuntimeError(f"Missing joints in URDF: {missing}")
    results, hinge, radius, handle_height = solve_keyframes(model)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_data(results)
    render(results, model, hinge, radius, handle_height)
    print("angle_deg palm_error_mm foot_error_mm cost success")
    for item in results:
        print(
            f'{math.degrees(item["angle"]):8.1f} '
            f'{1000*item["palm_error"]:13.3f} '
            f'{1000*item["foot_error"]:13.3f} '
            f'{item["cost"]:9.5f} {item["success"]}'
        )
    print(f"Wrote action visualizations to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
