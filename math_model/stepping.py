"""DS->SS->DS closed-chain stepping planner for refrigerator opening.

The planner works above a whole-body motion tracker. It plans contact mode,
step timing, landing pose, COM, pelvis, swing-foot and wrist references. It
never moves a stance foot and independently audits centroidal force and moment
balance under the refrigerator hand reaction.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .centroidal import solve_centroidal_equilibrium
from .horizon import minimum_jerk
from .quasistatic import ContactEquilibriumResult, FootPatch


class SupportMode(str, Enum):
    DOUBLE_SUPPORT = "double_support"
    LEFT_SUPPORT = "left_support"
    RIGHT_SUPPORT = "right_support"


@dataclass(frozen=True)
class FootPose:
    position: np.ndarray
    yaw: float
    label: str

    def rotation(self) -> np.ndarray:
        cosine, sine = np.cos(self.yaw), np.sin(self.yaw)
        return np.array(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
        )


@dataclass(frozen=True)
class StepPlannerConfig:
    swing_clearance_m: float = 0.065
    min_single_support_duration_s: float = 0.65
    foot_half_length_m: float = 0.12
    foot_half_width_m: float = 0.05
    friction_coefficient: float = 0.7
    min_support_margin_m: float = 0.004
    min_friction_margin_n: float = -1e-5
    comfortable_arm_reach_m: float = 0.32
    trigger_arm_reach_m: float = 0.37
    max_arm_reach_m: float = 0.42
    max_step_length_m: float = 0.34
    max_step_yaw_rad: float = np.deg2rad(25.0)
    min_lateral_separation_m: float = 0.12
    max_contact_audits: int = 12
    max_com_acceleration_m_s2: float = 3.5


@dataclass(frozen=True)
class StepPlanningProblem:
    time_s: np.ndarray
    door_angle_rad: np.ndarray
    hand_positions_w: np.ndarray
    hand_forces_on_robot_w: np.ndarray
    nominal_pelvis_positions_w: np.ndarray
    nominal_pelvis_yaw_rad: np.ndarray
    nominal_shoulder_positions_w: np.ndarray
    nominal_com_positions_w: np.ndarray
    left_foot_initial: FootPose
    right_foot_initial: FootPose
    mass_kg: float


@dataclass(frozen=True)
class StepCandidate:
    swing_foot: str
    lift_index: int
    land_index: int
    landing_position_w: np.ndarray
    landing_yaw_rad: float


@dataclass
class StepReference:
    candidate: StepCandidate
    support_modes: np.ndarray
    left_foot_positions_w: np.ndarray
    right_foot_positions_w: np.ndarray
    left_foot_yaw_rad: np.ndarray
    right_foot_yaw_rad: np.ndarray
    com_positions_w: np.ndarray
    pelvis_positions_w: np.ndarray
    pelvis_yaw_rad: np.ndarray
    shoulder_positions_w: np.ndarray
    arm_reach_m: np.ndarray
    com_acceleration_w: np.ndarray
    contact_results: list[ContactEquilibriumResult]
    metrics: dict[str, float | bool | str | list]


def _check_problem(problem: StepPlanningProblem) -> int:
    time_s = np.asarray(problem.time_s, dtype=float)
    frame_count = len(time_s)
    if frame_count < 7 or np.any(np.diff(time_s) <= 0.0):
        raise ValueError("time_s must be strictly increasing with at least 7 nodes")
    shapes = {
        "door_angle_rad": (frame_count,),
        "hand_positions_w": (frame_count, 3),
        "hand_forces_on_robot_w": (frame_count, 3),
        "nominal_pelvis_positions_w": (frame_count, 3),
        "nominal_pelvis_yaw_rad": (frame_count,),
        "nominal_shoulder_positions_w": (frame_count, 3),
        "nominal_com_positions_w": (frame_count, 3),
    }
    for name, shape in shapes.items():
        if np.asarray(getattr(problem, name)).shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    if problem.mass_kg <= 0.0:
        raise ValueError("mass_kg must be positive")
    return frame_count


def _yaw_interpolate(start: float, end: float, phase: np.ndarray) -> np.ndarray:
    delta = np.arctan2(np.sin(end - start), np.cos(end - start))
    return start + minimum_jerk(phase) * delta


def _segment(start: np.ndarray, end: np.ndarray, phase: np.ndarray) -> np.ndarray:
    return start + (end - start) * minimum_jerk(phase)[..., None]


def _acceleration(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    edge_order = 2 if len(time_s) >= 3 else 1
    velocity = np.gradient(values, time_s, axis=0, edge_order=edge_order)
    return np.gradient(velocity, time_s, axis=0, edge_order=edge_order)


def _foot_patch(
    position: np.ndarray,
    yaw: float,
    label: str,
    config: StepPlannerConfig,
) -> FootPatch:
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    return FootPatch(
        np.asarray(position, dtype=float),
        rotation,
        config.foot_half_length_m,
        config.foot_half_width_m,
        label,
    )


def detect_step_trigger(
    problem: StepPlanningProblem,
    config: StepPlannerConfig,
) -> int:
    """Return the first node at which fixed-support arm reach is uncomfortable."""
    _check_problem(problem)
    reach = np.linalg.norm(
        problem.hand_positions_w - problem.nominal_shoulder_positions_w,
        axis=1,
    )
    hit = np.flatnonzero(reach >= config.trigger_arm_reach_m)
    return int(hit[0]) if len(hit) else max(2, len(reach) // 2)


def generate_step_reference(
    problem: StepPlanningProblem,
    candidate: StepCandidate,
    config: StepPlannerConfig = StepPlannerConfig(),
    *,
    audit_contacts: bool = True,
) -> StepReference:
    """Generate and audit one explicit DS->SS->DS candidate."""
    frame_count = _check_problem(problem)
    lift = int(candidate.lift_index)
    land = int(candidate.land_index)
    if candidate.swing_foot not in {"left", "right"}:
        raise ValueError("swing_foot must be 'left' or 'right'")
    if not (1 <= lift < land <= frame_count - 2):
        raise ValueError("candidate indices must satisfy 1 <= lift < land <= N-2")
    if (
        problem.time_s[land] - problem.time_s[lift]
        < config.min_single_support_duration_s
    ):
        raise ValueError("single-support phase is too short")

    swing_initial = (
        problem.left_foot_initial
        if candidate.swing_foot == "left"
        else problem.right_foot_initial
    )
    stance_initial = (
        problem.right_foot_initial
        if candidate.swing_foot == "left"
        else problem.left_foot_initial
    )
    landing = np.asarray(candidate.landing_position_w, dtype=float)
    step_vector = landing[:2] - np.asarray(swing_initial.position, dtype=float)[:2]
    if np.linalg.norm(step_vector) > config.max_step_length_m + 1e-12:
        raise ValueError("step exceeds max_step_length_m")
    yaw_delta = np.arctan2(
        np.sin(candidate.landing_yaw_rad - swing_initial.yaw),
        np.cos(candidate.landing_yaw_rad - swing_initial.yaw),
    )
    if abs(yaw_delta) > config.max_step_yaw_rad + 1e-12:
        raise ValueError("step yaw exceeds max_step_yaw_rad")

    heading = np.array([np.cos(stance_initial.yaw), np.sin(stance_initial.yaw)])
    lateral_axis = np.array([-heading[1], heading[0]])
    signed_separation = float(
        np.dot(
            landing[:2] - np.asarray(stance_initial.position, dtype=float)[:2],
            lateral_axis,
        )
    )
    expected_sign = 1.0 if candidate.swing_foot == "left" else -1.0
    if expected_sign * signed_separation < config.min_lateral_separation_m:
        raise ValueError("landing pose crosses or narrows the stance excessively")

    left_positions = np.repeat(
        np.asarray(problem.left_foot_initial.position, dtype=float)[None, :],
        frame_count,
        axis=0,
    )
    right_positions = np.repeat(
        np.asarray(problem.right_foot_initial.position, dtype=float)[None, :],
        frame_count,
        axis=0,
    )
    left_yaw = np.full(frame_count, problem.left_foot_initial.yaw)
    right_yaw = np.full(frame_count, problem.right_foot_initial.yaw)

    swing_indices = np.arange(lift, land + 1)
    phase = (swing_indices - lift) / (land - lift)
    horizontal_phase = minimum_jerk(phase)
    swing_start = np.asarray(swing_initial.position, dtype=float)
    swing_positions = np.repeat(swing_start[None, :], len(swing_indices), axis=0)
    swing_positions[:, :2] = swing_start[:2] + (
        landing[:2] - swing_start[:2]
    ) * horizontal_phase[:, None]
    swing_positions[:, 2] = (
        swing_start[2]
        + (landing[2] - swing_start[2]) * horizontal_phase
        + config.swing_clearance_m * 64.0 * phase**3 * (1.0 - phase) ** 3
    )
    swing_yaw = _yaw_interpolate(
        swing_initial.yaw,
        candidate.landing_yaw_rad,
        phase,
    )
    if candidate.swing_foot == "left":
        left_positions[swing_indices] = swing_positions
        left_positions[land + 1 :] = landing
        left_yaw[swing_indices] = swing_yaw
        left_yaw[land + 1 :] = candidate.landing_yaw_rad
    else:
        right_positions[swing_indices] = swing_positions
        right_positions[land + 1 :] = landing
        right_yaw[swing_indices] = swing_yaw
        right_yaw[land + 1 :] = candidate.landing_yaw_rad

    modes = np.full(
        frame_count,
        SupportMode.DOUBLE_SUPPORT.value,
        dtype="<U16",
    )
    modes[lift + 1 : land] = (
        SupportMode.RIGHT_SUPPORT.value
        if candidate.swing_foot == "left"
        else SupportMode.LEFT_SUPPORT.value
    )

    initial_com = np.asarray(problem.nominal_com_positions_w[0], dtype=float)
    stance_xy = np.asarray(stance_initial.position, dtype=float)[:2]
    final_support_center = 0.5 * (
        np.asarray(stance_initial.position, dtype=float)[:2] + landing[:2]
    )
    com = np.zeros((frame_count, 3))
    com[:, 2] = problem.nominal_com_positions_w[:, 2]
    transfer_indices = np.arange(0, lift + 1)
    transfer_phase = transfer_indices / max(lift, 1)
    com[transfer_indices, :2] = _segment(
        initial_com[:2],
        stance_xy,
        transfer_phase,
    )
    com[lift : land + 1, :2] = stance_xy
    settle_indices = np.arange(land, frame_count)
    settle_phase = (settle_indices - land) / max(frame_count - 1 - land, 1)
    com[settle_indices, :2] = _segment(
        stance_xy,
        final_support_center,
        settle_phase,
    )

    initial_support_center = 0.5 * (
        np.asarray(problem.left_foot_initial.position, dtype=float)[:2]
        + np.asarray(problem.right_foot_initial.position, dtype=float)[:2]
    )
    support_shift = final_support_center - initial_support_center
    progress = np.zeros(frame_count)
    after_lift = np.arange(lift, frame_count)
    progress[after_lift] = minimum_jerk(
        (after_lift - lift) / max(frame_count - 1 - lift, 1)
    )
    pelvis = np.asarray(problem.nominal_pelvis_positions_w, dtype=float).copy()
    pelvis[:, :2] += progress[:, None] * support_shift
    final_heading = float(
        np.arctan2(
            np.sin(candidate.landing_yaw_rad),
            np.cos(candidate.landing_yaw_rad),
        )
    )
    heading_delta = np.arctan2(
        np.sin(final_heading - problem.nominal_pelvis_yaw_rad[-1]),
        np.cos(final_heading - problem.nominal_pelvis_yaw_rad[-1]),
    )
    pelvis_yaw = (
        np.asarray(problem.nominal_pelvis_yaw_rad, dtype=float)
        + progress * heading_delta
    )
    shoulder = np.asarray(problem.nominal_shoulder_positions_w, dtype=float).copy()
    shoulder[:, :2] += progress[:, None] * support_shift
    reach = np.linalg.norm(
        np.asarray(problem.hand_positions_w, dtype=float) - shoulder,
        axis=1,
    )
    com_acceleration = _acceleration(
        com,
        np.asarray(problem.time_s, dtype=float),
    )

    contact_results: list[ContactEquilibriumResult] = []
    if audit_contacts:
        for frame, mode in enumerate(modes):
            active_feet: list[FootPatch] = []
            if mode in {
                SupportMode.DOUBLE_SUPPORT.value,
                SupportMode.LEFT_SUPPORT.value,
            }:
                active_feet.append(
                    _foot_patch(
                        left_positions[frame],
                        left_yaw[frame],
                        "left",
                        config,
                    )
                )
            if mode in {
                SupportMode.DOUBLE_SUPPORT.value,
                SupportMode.RIGHT_SUPPORT.value,
            }:
                active_feet.append(
                    _foot_patch(
                        right_positions[frame],
                        right_yaw[frame],
                        "right",
                        config,
                    )
                )
            contact_results.append(
                solve_centroidal_equilibrium(
                    mass=problem.mass_kg,
                    center_of_mass=com[frame],
                    com_acceleration=com_acceleration[frame],
                    angular_momentum_rate=np.zeros(3),
                    hand_point=problem.hand_positions_w[frame],
                    hand_force_on_robot=problem.hand_forces_on_robot_w[frame],
                    feet=active_feet,
                    friction_coefficient=config.friction_coefficient,
                )
            )

    stance_positions = (
        right_positions if candidate.swing_foot == "left" else left_positions
    )
    stance_origin = np.asarray(stance_initial.position, dtype=float)
    stance_drift = float(
        np.max(np.linalg.norm(stance_positions - stance_origin, axis=1))
    )
    realized_swing_positions = (
        left_positions if candidate.swing_foot == "left" else right_positions
    )
    peak_clearance = float(
        np.max(realized_swing_positions[lift : land + 1, 2] - swing_start[2])
    )
    minimum_support = float(
        min((result.support_margin for result in contact_results), default=np.inf)
    )
    minimum_friction = float(
        min(
            (result.minimum_friction_margin for result in contact_results),
            default=np.inf,
        )
    )
    contact_fraction = (
        float(np.mean([result.feasible for result in contact_results]))
        if contact_results
        else 1.0
    )
    maximum_acceleration = float(
        np.max(np.linalg.norm(com_acceleration, axis=1))
    )
    fixed_support_maximum_reach = float(
        np.max(
            np.linalg.norm(
                problem.hand_positions_w - problem.nominal_shoulder_positions_w,
                axis=1,
            )
        )
    )
    maximum_reach = float(np.max(reach))

    accepted = bool(
        stance_drift <= 1e-9
        and peak_clearance >= 0.95 * config.swing_clearance_m
        and maximum_reach <= config.max_arm_reach_m
        and maximum_acceleration <= config.max_com_acceleration_m_s2
        and contact_fraction == 1.0
        and minimum_support >= config.min_support_margin_m
        and minimum_friction >= config.min_friction_margin_n
    )
    failure_reasons: list[str] = []
    if stance_drift > 1e-9:
        failure_reasons.append("stance_foot_drift")
    if peak_clearance < 0.95 * config.swing_clearance_m:
        failure_reasons.append("swing_clearance")
    if maximum_reach > config.max_arm_reach_m:
        failure_reasons.append("arm_reach")
    if maximum_acceleration > config.max_com_acceleration_m_s2:
        failure_reasons.append("com_acceleration")
    if contact_fraction < 1.0:
        failure_reasons.append("centroidal_contact_feasibility")
    if minimum_support < config.min_support_margin_m:
        failure_reasons.append("support_margin")
    if minimum_friction < config.min_friction_margin_n:
        failure_reasons.append("friction_margin")

    metrics: dict[str, float | bool | str | list] = {
        "accepted": accepted,
        "failure_reasons": failure_reasons,
        "swing_foot": candidate.swing_foot,
        "step_length_m": float(np.linalg.norm(step_vector)),
        "peak_swing_clearance_m": peak_clearance,
        "maximum_stance_foot_drift_m": stance_drift,
        "maximum_arm_reach_m": maximum_reach,
        "fixed_support_maximum_arm_reach_m": fixed_support_maximum_reach,
        "arm_reach_improvement_m": fixed_support_maximum_reach - maximum_reach,
        "maximum_com_acceleration_m_s2": maximum_acceleration,
        "minimum_support_margin_m": minimum_support,
        "minimum_friction_margin_n": minimum_friction,
        "contact_feasible_fraction": contact_fraction,
    }
    return StepReference(
        candidate,
        modes,
        left_positions,
        right_positions,
        left_yaw,
        right_yaw,
        com,
        pelvis,
        pelvis_yaw,
        shoulder,
        reach,
        com_acceleration,
        contact_results,
        metrics,
    )


def _candidate_score(
    reference: StepReference,
    config: StepPlannerConfig,
) -> float:
    metrics = reference.metrics
    penalty = 0.0
    penalty += 4e4 * max(
        0.0,
        float(metrics["maximum_arm_reach_m"]) - config.comfortable_arm_reach_m,
    ) ** 2
    penalty += 30.0 * float(metrics["step_length_m"]) ** 2
    penalty += 0.4 * float(metrics["maximum_com_acceleration_m_s2"]) ** 2
    if not metrics["accepted"]:
        penalty += 1e5 + 2e4 * (
            1.0 - float(metrics["contact_feasible_fraction"])
        )
        penalty += 2e4 * max(
            0.0,
            config.min_support_margin_m
            - float(metrics["minimum_support_margin_m"]),
        )
    return float(penalty)


def _kinematic_candidate_score(
    problem: StepPlanningProblem,
    candidate: StepCandidate,
    config: StepPlannerConfig,
) -> float:
    reference = generate_step_reference(
        problem,
        candidate,
        config,
        audit_contacts=False,
    )
    return _candidate_score(reference, config)


def candidate_set(
    problem: StepPlanningProblem,
    config: StepPlannerConfig = StepPlannerConfig(),
    *,
    trigger_index: int | None = None,
) -> list[StepCandidate]:
    """Enumerate left/right step timing and landing candidates."""
    frame_count = _check_problem(problem)
    trigger = (
        detect_step_trigger(problem, config)
        if trigger_index is None
        else int(trigger_index)
    )
    trigger = int(np.clip(trigger, 2, frame_count - 3))
    dt = float(np.median(np.diff(problem.time_s)))
    minimum_nodes = max(
        2,
        int(np.ceil(config.min_single_support_duration_s / dt)),
    )
    lift_options = sorted(
        {
            int(np.clip(trigger - delta, 1, frame_count - 2 - minimum_nodes))
            for delta in (max(2, frame_count // 8), max(1, frame_count // 12))
        }
    )
    land_options = sorted(
        {
            int(
                np.clip(
                    trigger + delta,
                    min(lift_options) + minimum_nodes,
                    frame_count - 2,
                )
            )
            for delta in (max(2, frame_count // 10), max(3, frame_count // 7))
        }
    )

    final_shoulder = problem.nominal_shoulder_positions_w[-1]
    final_hand = problem.hand_positions_w[-1]
    horizontal_delta = final_hand[:2] - final_shoulder[:2]
    final_distance = float(np.linalg.norm(final_hand - final_shoulder))
    horizontal_norm = max(float(np.linalg.norm(horizontal_delta)), 1e-9)
    direction = horizontal_delta / horizontal_norm
    excess = max(
        0.06,
        final_distance - config.comfortable_arm_reach_m + 0.035,
    )
    desired_shift = direction * min(excess, 0.82 * config.max_step_length_m)
    perpendicular = np.array([-direction[1], direction[0]])
    desired_heading = float(np.arctan2(direction[1], direction[0]))

    candidates: list[StepCandidate] = []
    for swing_foot in ("left", "right"):
        swing_initial = (
            problem.left_foot_initial
            if swing_foot == "left"
            else problem.right_foot_initial
        )
        stance_initial = (
            problem.right_foot_initial
            if swing_foot == "left"
            else problem.left_foot_initial
        )
        for lift in lift_options:
            for land in land_options:
                if land - lift < minimum_nodes:
                    continue
                for scale in (0.75, 1.0, 1.15):
                    for lateral in (-0.035, 0.0, 0.035):
                        shift = scale * desired_shift + lateral * perpendicular
                        shift_norm = float(np.linalg.norm(shift))
                        if shift_norm > config.max_step_length_m:
                            shift *= config.max_step_length_m / shift_norm
                        landing = np.asarray(
                            swing_initial.position,
                            dtype=float,
                        ).copy()
                        landing[:2] += shift

                        stance_heading = np.array(
                            [np.cos(stance_initial.yaw), np.sin(stance_initial.yaw)]
                        )
                        stance_lateral = np.array(
                            [-stance_heading[1], stance_heading[0]]
                        )
                        signed_separation = float(
                            np.dot(
                                landing[:2]
                                - np.asarray(stance_initial.position, dtype=float)[:2],
                                stance_lateral,
                            )
                        )
                        expected_sign = 1.0 if swing_foot == "left" else -1.0
                        correction = max(
                            0.0,
                            config.min_lateral_separation_m
                            - expected_sign * signed_separation,
                        )
                        landing[:2] += (
                            expected_sign * correction * stance_lateral
                        )

                        desired_yaw_delta = np.clip(
                            np.arctan2(
                                np.sin(desired_heading - swing_initial.yaw),
                                np.cos(desired_heading - swing_initial.yaw),
                            ),
                            -config.max_step_yaw_rad,
                            config.max_step_yaw_rad,
                        )
                        candidates.append(
                            StepCandidate(
                                swing_foot,
                                lift,
                                land,
                                landing,
                                swing_initial.yaw + 0.45 * desired_yaw_delta,
                            )
                        )
    return candidates


def plan_best_single_step(
    problem: StepPlanningProblem,
    config: StepPlannerConfig = StepPlannerConfig(),
    *,
    trigger_index: int | None = None,
) -> StepReference:
    """Select the lowest-cost feasible left- or right-foot step."""
    candidates = candidate_set(problem, config, trigger_index=trigger_index)
    ranked = sorted(
        candidates,
        key=lambda candidate: _kinematic_candidate_score(
            problem,
            candidate,
            config,
        ),
    )
    audited = [
        generate_step_reference(problem, candidate, config, audit_contacts=True)
        for candidate in ranked[: max(1, config.max_contact_audits)]
    ]
    accepted = [reference for reference in audited if reference.metrics["accepted"]]
    pool = accepted if accepted else audited
    best = min(pool, key=lambda reference: _candidate_score(reference, config))
    best.metrics["evaluated_candidates"] = len(candidates)
    best.metrics["contact_audited_candidates"] = len(audited)
    return best
