"""Centroidal contact-force feasibility with prescribed COM acceleration.

This extends the quasi-static support audit to DS/SS phases without changing the
legacy quasi-static API. At each node it solves

    m c_ddot = sum(f_i) + f_hand + m g
    L_dot     = sum((p_i-c) x f_i) + (p_hand-c) x f_hand

with unilateral sole-corner forces and a conservative friction pyramid.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.spatial import ConvexHull

from .quasistatic import ContactEquilibriumResult, FootPatch


def _support_margin(point_xy: np.ndarray, support_xy: np.ndarray) -> float:
    hull = ConvexHull(np.asarray(support_xy, dtype=float))
    normal = hull.equations[:, :2]
    offset = hull.equations[:, 2]
    signed = -(normal @ point_xy + offset) / np.linalg.norm(normal, axis=1)
    return float(np.min(signed))


def solve_centroidal_equilibrium(
    *,
    mass: float,
    center_of_mass: np.ndarray,
    com_acceleration: np.ndarray,
    angular_momentum_rate: np.ndarray,
    hand_point: np.ndarray,
    hand_force_on_robot: np.ndarray,
    feet: Sequence[FootPatch],
    friction_coefficient: float = 0.7,
    gravity: float = 9.81,
    force_tolerance: float = 1e-5,
    moment_tolerance: float = 1e-5,
    max_iterations: int = 1000,
) -> ContactEquilibriumResult:
    """Solve contact forces for a prescribed centroidal state.

    A single active foot represents SS; two active feet represent DS. Swing-foot
    forces are absent by construction rather than being softly penalized.
    """
    if mass <= 0.0 or friction_coefficient <= 0.0 or not feet:
        raise ValueError("invalid mass, friction, or support set")

    com = np.asarray(center_of_mass, dtype=float)
    acceleration = np.asarray(com_acceleration, dtype=float)
    momentum_rate = np.asarray(angular_momentum_rate, dtype=float)
    hand = np.asarray(hand_point, dtype=float)
    hand_force = np.asarray(hand_force_on_robot, dtype=float)
    for name, value in (
        ("center_of_mass", com),
        ("com_acceleration", acceleration),
        ("angular_momentum_rate", momentum_rate),
        ("hand_point", hand),
        ("hand_force_on_robot", hand_force),
    ):
        if value.shape != (3,):
            raise ValueError(f"{name} must be a 3-vector")

    points = np.concatenate([foot.corners() for foot in feet], axis=0)
    count = len(points)
    variable_count = 3 * count
    gravity_force = np.array([0.0, 0.0, -mass * gravity])
    hand_moment = np.cross(hand - com, hand_force)

    a_eq = np.zeros((6, variable_count))
    for index, point in enumerate(points):
        section = slice(3 * index, 3 * index + 3)
        a_eq[:3, section] = np.eye(3)
        rx, ry, rz = point - com
        a_eq[3:, section] = np.array(
            [[0.0, -rz, ry], [rz, 0.0, -rx], [-ry, rx, 0.0]]
        )
    b_eq = np.r_[
        mass * acceleration - hand_force - gravity_force,
        momentum_rate - hand_moment,
    ]

    pyramid_mu = friction_coefficient / np.sqrt(2.0)
    a_ub = np.zeros((4 * count, variable_count))
    for index in range(count):
        base = 4 * index
        fx, fy, fz = 3 * index, 3 * index + 1, 3 * index + 2
        a_ub[base + 0, [fx, fz]] = [1.0, -pyramid_mu]
        a_ub[base + 1, [fx, fz]] = [-1.0, -pyramid_mu]
        a_ub[base + 2, [fy, fz]] = [1.0, -pyramid_mu]
        a_ub[base + 3, [fy, fz]] = [-1.0, -pyramid_mu]
    b_ub = np.zeros(4 * count)
    bounds = [(None, None), (None, None), (0.0, None)] * count

    feasibility = linprog(
        np.zeros(variable_count),
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if feasibility.success:
        initial = feasibility.x

        def objective(flat: np.ndarray) -> float:
            forces = flat.reshape(count, 3)
            concentration = forces - forces.mean(axis=0)
            return float(
                0.5 * np.sum(concentration**2)
                + 1e-4 * np.sum(forces**2)
                + 1e-3 * np.sum(forces[:, :2] ** 2)
            )

        constraints = [
            {"type": "eq", "fun": lambda flat: a_eq @ flat - b_eq},
            {"type": "ineq", "fun": lambda flat: b_ub - a_ub @ flat},
        ]
        refined = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": int(max_iterations), "ftol": 1e-12, "disp": False},
        )
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
        cop = np.full(3, np.nan)
        support_margin = float("-inf")

    force_residual = residual[:3]
    moment_residual = residual[3:]
    feasible = bool(
        feasibility.success
        and np.linalg.norm(force_residual) <= force_tolerance
        and np.linalg.norm(moment_residual) <= moment_tolerance
        and np.min(margins) >= -force_tolerance
        and support_margin >= -force_tolerance
    )
    return ContactEquilibriumResult(
        feasible,
        optimizer_success,
        message,
        points,
        forces,
        force_residual,
        moment_residual,
        margins,
        cop,
        support_margin,
        objective_value,
    )
