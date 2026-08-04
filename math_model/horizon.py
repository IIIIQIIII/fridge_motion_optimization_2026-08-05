"""Reusable full-horizon trajectory objective terms.

The previous project optimized one door angle at a time and repaired bad IK
branches after the fact.  These terms operate on the complete state sequence,
so early configurations are chosen with later reachability and smoothness in
view.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ManipulabilityMetrics:
    sigma_min: float
    sigma_max: float
    condition_number: float
    log_volume: float


def minimum_jerk(value: float | np.ndarray) -> float | np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return 10.0 * clipped**3 - 15.0 * clipped**4 + 6.0 * clipped**5


def difference_residuals(
    states: np.ndarray,
    dt: float,
    *,
    velocity_weight: float = 0.0,
    acceleration_weight: float = 1.0,
    jerk_weight: float = 1.0,
) -> np.ndarray:
    values = np.asarray(states, dtype=float)
    if values.ndim != 2:
        raise ValueError("states must have shape (frames, variables)")
    if values.shape[0] < 2:
        return np.empty(0, dtype=float)
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    terms: list[np.ndarray] = []
    velocity = np.diff(values, axis=0) / dt
    if velocity_weight:
        terms.append(float(velocity_weight) * velocity.ravel())

    if values.shape[0] >= 3:
        acceleration = np.diff(velocity, axis=0) / dt
        if acceleration_weight:
            terms.append(float(acceleration_weight) * acceleration.ravel())
    else:
        acceleration = np.empty((0, values.shape[1]))

    if values.shape[0] >= 4 and jerk_weight:
        jerk = np.diff(acceleration, axis=0) / dt
        terms.append(float(jerk_weight) * jerk.ravel())

    return np.concatenate(terms) if terms else np.empty(0, dtype=float)


def joint_limit_margin(configuration: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    q = np.asarray(configuration, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if q.shape != lo.shape or q.shape != hi.shape:
        raise ValueError("configuration and limit arrays must have equal shapes")
    span = hi - lo
    if np.any(span <= 0.0):
        raise ValueError("every upper limit must exceed its lower limit")
    return np.minimum((q - lo) / span, (hi - q) / span)


def joint_limit_barrier(
    configuration: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    desired_margin: float = 0.12,
    sharpness: float = 35.0,
) -> np.ndarray:
    if desired_margin <= 0.0 or desired_margin >= 0.5:
        raise ValueError("desired_margin must lie in (0, 0.5)")
    if sharpness <= 0.0:
        raise ValueError("sharpness must be positive")
    margin = joint_limit_margin(configuration, lower, upper)
    scaled = sharpness * (desired_margin - margin)
    return (np.maximum(scaled, 0.0) + np.log1p(np.exp(-np.abs(scaled)))) / sharpness


def manipulability_metrics(jacobian: np.ndarray, *, epsilon: float = 1e-12) -> ManipulabilityMetrics:
    matrix = np.asarray(jacobian, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("jacobian must be a non-empty matrix")
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    sigma_max = float(singular_values[0])
    sigma_min = float(singular_values[-1])
    condition = sigma_max / max(sigma_min, epsilon)
    log_volume = float(np.sum(np.log(np.maximum(singular_values, epsilon))))
    return ManipulabilityMetrics(sigma_min, sigma_max, condition, log_volume)
