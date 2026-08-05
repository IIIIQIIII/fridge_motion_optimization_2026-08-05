"""Mathematical models for G1 refrigerator-opening motion optimization."""

from .centroidal import solve_centroidal_equilibrium
from .horizon import (
    difference_residuals,
    joint_limit_barrier,
    joint_limit_margin,
    manipulability_metrics,
    minimum_jerk,
)
from .quasistatic import (
    ContactEquilibriumResult,
    DoorResistanceModel,
    FootPatch,
    fit_hinge_circle,
    required_tangential_hand_force,
    solve_contact_equilibrium,
)
from .stepping import (
    FootPose,
    StepCandidate,
    StepPlannerConfig,
    StepPlanningProblem,
    StepReference,
    SupportMode,
    candidate_set,
    detect_step_trigger,
    generate_step_reference,
    plan_best_single_step,
)

__all__ = [
    "ContactEquilibriumResult",
    "DoorResistanceModel",
    "FootPatch",
    "FootPose",
    "StepCandidate",
    "StepPlannerConfig",
    "StepPlanningProblem",
    "StepReference",
    "SupportMode",
    "candidate_set",
    "detect_step_trigger",
    "difference_residuals",
    "fit_hinge_circle",
    "generate_step_reference",
    "joint_limit_barrier",
    "joint_limit_margin",
    "manipulability_metrics",
    "minimum_jerk",
    "plan_best_single_step",
    "required_tangential_hand_force",
    "solve_centroidal_equilibrium",
    "solve_contact_equilibrium",
]
