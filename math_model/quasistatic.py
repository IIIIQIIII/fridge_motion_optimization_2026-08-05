"""Quasi-static door, grasp, and support models for humanoid manipulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.spatial import ConvexHull

_GRAVITY = 9.81


@dataclass(frozen=True)
class DoorResistanceModel:
    breakaway_torque: float = 6.0
    coulomb_torque: float = 3.0
    spring_torque_per_rad: float = 0.5
    viscous_torque_per_rad_s: float = 0.2
    breakaway_angle_rad: float = np.deg2rad(2.0)

    def torque(self, angle_rad: float, angular_speed_rad_s: float = 0.0) -> float:
        angle = abs(float(angle_rad))
        speed = abs(float(angular_speed_rad_s))
        base = self.breakaway_torque if angle <= self.breakaway_angle_rad else self.coulomb_torque
        return float(base + self.spring_torque_per_rad * angle + self.viscous_torque_per_rad_s * speed)


@dataclass(frozen=True)
class FootPatch:
    center: np.ndarray
    rotation: np.ndarray
    half_length: float
    half_width: float
    label: str

    def corners(self) -> np.ndarray:
        center = np.asarray(self.center, dtype=float)
        rotation = np.asarray(self.rotation, dtype=float)
        if center.shape != (3,) or rotation.shape != (3, 3):
            raise ValueError("center must be (3,) and rotation must be (3, 3)")
        if self.half_length <= 0.0 or self.half_width <= 0.0:
            raise ValueError("foot half dimensions must be positive")
        local = np.array([
            [self.half_length, self.half_width, 0.0],
            [self.half_length, -self.half_width, 0.0],
            [-self.half_length, self.half_width, 0.0],
            [-self.half_length, -self.half_width, 0.0],
        ])
        return local @ rotation.T + center


@dataclass(frozen=True)
class HingeCircle:
    center: np.ndarray
    axis: np.ndarray
    radius: float
    rms_error: float


@dataclass(frozen=True)
class ContactEquilibriumResult:
    feasible: bool
    optimizer_success: bool
    message: str
    contact_points: np.ndarray
    contact_forces: np.ndarray
    force_residual: np.ndarray
    moment_residual: np.ndarray
    friction_margins: np.ndarray
    center_of_pressure: np.ndarray
    support_margin: float
    objective: float

    @property
    def minimum_friction_margin(self) -> float:
        return float(np.min(self.friction_margins))


def _normalize(vector: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if value.shape != (3,) or norm <= 1e-12:
        raise ValueError(f"{name} must be a non-zero 3-vector")
    return value / norm


def fit_hinge_circle(points: np.ndarray) -> HingeCircle:
    samples = np.asarray(points, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 3 or samples.shape[0] < 3:
        raise ValueError("points must have shape (N, 3) with N >= 3")
    mean = samples.mean(axis=0)
    _, _, vh = np.linalg.svd(samples - mean, full_matrices=False)
    axis, u = vh[-1], vh[0]
    v = np.cross(axis, u)
    v /= np.linalg.norm(v)
    coords = np.column_stack(((samples - mean) @ u, (samples - mean) @ v))
    matrix = np.column_stack((2.0 * coords[:, 0], 2.0 * coords[:, 1], np.ones(len(coords))))
    rhs = np.sum(coords**2, axis=1)
    cx, cy, c0 = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    radius_sq = c0 + cx**2 + cy**2
    if radius_sq <= 0.0:
        raise ValueError("fitted circle has non-positive radius")
    radius = float(np.sqrt(radius_sq))
    center = mean + cx * u + cy * v
    radial0, radial1 = samples[0] - center, samples[1] - center
    if np.dot(np.cross(radial0, radial1), axis) < 0.0:
        axis = -axis
    delta = samples - center
    radial_distance = np.linalg.norm(delta - np.outer(delta @ axis, axis), axis=1)
    plane_distance = delta @ axis
    rms = float(np.sqrt(np.mean((radial_distance - radius) ** 2 + plane_distance**2)))
    return HingeCircle(center, axis, radius, rms)


def required_tangential_hand_force(
    hinge_point: np.ndarray,
    hinge_axis: np.ndarray,
    handle_point: np.ndarray,
    opening_torque: float,
) -> np.ndarray:
    axis = _normalize(hinge_axis, name="hinge_axis")
    hinge, handle = np.asarray(hinge_point, dtype=float), np.asarray(handle_point, dtype=float)
    radial = handle - hinge
    radial -= axis * np.dot(radial, axis)
    radius = float(np.linalg.norm(radial))
    if radius <= 1e-9:
        raise ValueError("handle lies on the hinge axis")
    tangent = np.cross(axis, radial / radius)
    return -(float(opening_torque) / radius) * tangent


def _support_margin(point_xy: np.ndarray, support_xy: np.ndarray) -> float:
    hull = ConvexHull(np.asarray(support_xy, dtype=float))
    normal, offset = hull.equations[:, :2], hull.equations[:, 2]
    signed = -(normal @ point_xy + offset) / np.linalg.norm(normal, axis=1)
    return float(np.min(signed))


def solve_contact_equilibrium(
    *,
    mass: float,
    center_of_mass: np.ndarray,
    hand_point: np.ndarray,
    hand_force_on_robot: np.ndarray,
    feet: Sequence[FootPatch],
    friction_coefficient: float = 0.7,
    gravity: float = _GRAVITY,
    force_tolerance: float = 1e-5,
    moment_tolerance: float = 1e-5,
    max_iterations: int = 1000,
) -> ContactEquilibriumResult:
    if mass <= 0.0 or friction_coefficient <= 0.0 or not feet:
        raise ValueError("invalid mass, friction, or support set")
    com = np.asarray(center_of_mass, dtype=float)
    hand = np.asarray(hand_point, dtype=float)
    hand_force = np.asarray(hand_force_on_robot, dtype=float)
    points = np.concatenate([foot.corners() for foot in feet], axis=0)
    count, variables = len(points), 3 * len(points)
    gravity_force = np.array([0.0, 0.0, -mass * gravity])
    hand_moment = np.cross(hand - com, hand_force)

    a_eq = np.zeros((6, variables))
    for index, point in enumerate(points):
        sl = slice(3 * index, 3 * index + 3)
        a_eq[:3, sl] = np.eye(3)
        rx, ry, rz = point - com
        a_eq[3:, sl] = np.array([[0.0, -rz, ry], [rz, 0.0, -rx], [-ry, rx, 0.0]])
    b_eq = -np.r_[hand_force + gravity_force, hand_moment]

    pyramid_mu = friction_coefficient / np.sqrt(2.0)
    a_ub = np.zeros((4 * count, variables))
    for index in range(count):
        base = 4 * index
        fx, fy, fz = 3 * index, 3 * index + 1, 3 * index + 2
        a_ub[base + 0, [fx, fz]] = [1.0, -pyramid_mu]
        a_ub[base + 1, [fx, fz]] = [-1.0, -pyramid_mu]
        a_ub[base + 2, [fy, fz]] = [1.0, -pyramid_mu]
        a_ub[base + 3, [fy, fz]] = [-1.0, -pyramid_mu]
    b_ub = np.zeros(4 * count)
    bounds = [(None, None), (None, None), (0.0, None)] * count
    feasibility = linprog(np.zeros(variables), A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if feasibility.success:
        initial = feasibility.x
        def objective(flat: np.ndarray) -> float:
            forces = flat.reshape(count, 3)
            concentration = forces - forces.mean(axis=0)
            return float(0.5 * np.sum(concentration**2) + 1e-4 * np.sum(forces**2) + 1e-3 * np.sum(forces[:, :2]**2))
        constraints = [
            {"type": "eq", "fun": lambda flat: a_eq @ flat - b_eq},
            {"type": "ineq", "fun": lambda flat: b_ub - a_ub @ flat},
        ]
        refined = minimize(objective, initial, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": int(max_iterations), "ftol": 1e-12, "disp": False})
        flat = refined.x if refined.success else initial
        optimizer_success = bool(refined.success or feasibility.success)
        message = str(refined.message if refined.success else feasibility.message)
        objective_value = objective(flat)
    else:
        flat = np.linalg.lstsq(a_eq, b_eq, rcond=None)[0]
        optimizer_success = False
        message = str(feasibility.message)
        objective_value = float(np.sum(flat**2))

    forces = flat.reshape(count, 3)
    residual = a_eq @ flat - b_eq
    tangential = np.linalg.norm(forces[:, :2], axis=1)
    margins = friction_coefficient * forces[:, 2] - tangential
    total_normal = float(np.sum(forces[:, 2]))
    if total_normal > 1e-10:
        cop = np.sum(points * forces[:, 2, None], axis=0) / total_normal
        support_margin = _support_margin(cop[:2], points[:, :2])
    else:
        cop, support_margin = np.full(3, np.nan), float("-inf")
    force_residual, moment_residual = residual[:3], residual[3:]
    feasible = bool(feasibility.success and np.linalg.norm(force_residual) <= force_tolerance and np.linalg.norm(moment_residual) <= moment_tolerance and np.min(margins) >= -force_tolerance and support_margin >= -force_tolerance)
    return ContactEquilibriumResult(feasible, optimizer_success, message, points, forces, force_residual, moment_residual, margins, cop, support_margin, objective_value)
