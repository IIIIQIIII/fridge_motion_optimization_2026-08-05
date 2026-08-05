from __future__ import annotations

import numpy as np

from math_model.stepping import (
    FootPose,
    StepCandidate,
    StepPlannerConfig,
    StepPlanningProblem,
    SupportMode,
    generate_step_reference,
    plan_best_single_step,
)


def make_problem(
    *,
    nodes: int = 31,
    duration: float = 6.0,
    final_hand_x: float = 0.53,
    hand_force_y: float = 0.0,
    mass: float = 45.0,
) -> StepPlanningProblem:
    time_s = np.linspace(0.0, duration, nodes)
    angles = np.linspace(0.0, np.deg2rad(60.0), nodes)
    hand = np.column_stack(
        (
            np.linspace(0.30, final_hand_x, nodes),
            np.full(nodes, -0.02),
            np.full(nodes, 1.10),
        )
    )
    forces = np.zeros((nodes, 3))
    forces[:, 1] = hand_force_y
    pelvis = np.tile(np.array([0.0, 0.0, 0.82]), (nodes, 1))
    shoulder = np.tile(np.array([0.0, -0.16, 1.10]), (nodes, 1))
    center_of_mass = np.tile(np.array([0.0, 0.0, 0.80]), (nodes, 1))
    return StepPlanningProblem(
        time_s=time_s,
        door_angle_rad=angles,
        hand_positions_w=hand,
        hand_forces_on_robot_w=forces,
        nominal_pelvis_positions_w=pelvis,
        nominal_pelvis_yaw_rad=np.zeros(nodes),
        nominal_shoulder_positions_w=shoulder,
        nominal_com_positions_w=center_of_mass,
        left_foot_initial=FootPose(np.array([0.0, 0.10, 0.0]), 0.0, "left"),
        right_foot_initial=FootPose(np.array([0.0, -0.10, 0.0]), 0.0, "right"),
        mass_kg=mass,
    )


def test_ds_ss_ds_has_clearance_and_zero_stance_drift() -> None:
    problem = make_problem()
    config = StepPlannerConfig(
        min_support_margin_m=-1.0,
        max_arm_reach_m=1.0,
        max_com_acceleration_m_s2=10.0,
    )
    candidate = StepCandidate(
        "right",
        9,
        20,
        np.array([0.20, -0.10, 0.0]),
        0.0,
    )
    reference = generate_step_reference(
        problem,
        candidate,
        config,
        audit_contacts=False,
    )
    assert reference.support_modes[0] == SupportMode.DOUBLE_SUPPORT.value
    assert np.all(
        reference.support_modes[10:20] == SupportMode.LEFT_SUPPORT.value
    )
    assert reference.support_modes[20] == SupportMode.DOUBLE_SUPPORT.value
    assert (
        np.max(
            np.linalg.norm(
                reference.left_foot_positions_w
                - problem.left_foot_initial.position,
                axis=1,
            )
        )
        < 1e-12
    )
    assert (
        reference.metrics["peak_swing_clearance_m"]
        >= 0.95 * config.swing_clearance_m
    )
    assert np.allclose(
        reference.right_foot_positions_w[-1],
        candidate.landing_position_w,
    )


def test_single_support_centroidal_audit_is_feasible_for_slow_transfer() -> None:
    problem = make_problem(duration=9.0, final_hand_x=0.45)
    config = StepPlannerConfig(
        min_support_margin_m=0.0,
        max_arm_reach_m=0.8,
        max_com_acceleration_m_s2=4.0,
    )
    candidate = StepCandidate(
        "right",
        10,
        20,
        np.array([0.16, -0.10, 0.0]),
        0.0,
    )
    reference = generate_step_reference(problem, candidate, config)
    assert reference.metrics["contact_feasible_fraction"] == 1.0
    assert reference.metrics["minimum_support_margin_m"] >= -1e-7
    assert reference.metrics["maximum_stance_foot_drift_m"] < 1e-12


def test_planner_reduces_late_arm_extension() -> None:
    problem = make_problem(duration=9.0, final_hand_x=0.60)
    config = StepPlannerConfig(
        min_support_margin_m=0.0,
        max_arm_reach_m=0.49,
        trigger_arm_reach_m=0.40,
        max_contact_audits=18,
        max_com_acceleration_m_s2=4.0,
    )
    plan = plan_best_single_step(problem, config)
    assert plan.metrics["arm_reach_improvement_m"] > 0.06
    assert plan.metrics["maximum_stance_foot_drift_m"] < 1e-12
    assert plan.metrics["peak_swing_clearance_m"] >= 0.95 * config.swing_clearance_m
    assert plan.metrics["contact_feasible_fraction"] == 1.0


def test_large_hand_reaction_is_reported_infeasible_in_single_support() -> None:
    problem = make_problem(
        duration=6.0,
        final_hand_x=0.48,
        hand_force_y=500.0,
    )
    config = StepPlannerConfig(
        min_support_margin_m=0.0,
        max_arm_reach_m=1.0,
        max_com_acceleration_m_s2=10.0,
    )
    candidate = StepCandidate(
        "right",
        9,
        20,
        np.array([0.18, -0.10, 0.0]),
        0.0,
    )
    reference = generate_step_reference(problem, candidate, config)
    assert reference.metrics["contact_feasible_fraction"] < 1.0
    assert (
        "centroidal_contact_feasibility"
        in reference.metrics["failure_reasons"]
    )
