import numpy as np

from math_model.quasistatic import FootPatch, fit_hinge_circle, required_tangential_hand_force, solve_contact_equilibrium


def feet():
    return [
        FootPatch(np.array([0.0, 0.10, 0.0]), np.eye(3), 0.12, 0.05, "left"),
        FootPatch(np.array([0.0, -0.10, 0.0]), np.eye(3), 0.12, 0.05, "right"),
    ]


def test_fit_hinge_circle_recovers_geometry():
    center = np.array([1.0, 2.0, 0.8])
    radius = 0.45
    angles = np.linspace(0.0, np.pi / 3.0, 9)
    points = np.column_stack([center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles), np.full_like(angles, center[2])])
    fit = fit_hinge_circle(points)
    assert np.allclose(fit.center, center, atol=1e-9)
    assert np.isclose(fit.radius, radius, atol=1e-9)
    assert fit.rms_error < 1e-9


def test_required_force_has_correct_torque_and_reaction_sign():
    hinge = np.zeros(3)
    axis = np.array([0.0, 0.0, 1.0])
    handle = np.array([0.5, 0.0, 0.0])
    force_on_robot = required_tangential_hand_force(hinge, axis, handle, opening_torque=5.0)
    torque = np.dot(axis, np.cross(handle - hinge, -force_on_robot))
    assert np.isclose(torque, 5.0)
    assert np.isclose(np.linalg.norm(force_on_robot), 10.0)


def test_symmetric_standing_equilibrium_is_feasible():
    result = solve_contact_equilibrium(
        mass=50.0,
        center_of_mass=np.array([0.0, 0.0, 0.8]),
        hand_point=np.array([0.4, 0.0, 1.0]),
        hand_force_on_robot=np.zeros(3),
        feet=feet(),
        friction_coefficient=0.7,
    )
    assert result.feasible, result.message
    assert np.linalg.norm(result.force_residual) < 1e-5
    assert np.linalg.norm(result.moment_residual) < 1e-5
    assert result.support_margin > 0.0
    assert np.isclose(result.contact_forces[:, 2].sum(), 50.0 * 9.81, atol=1e-5)


def test_large_horizontal_reaction_exposes_low_friction_infeasibility():
    common = dict(
        mass=50.0,
        center_of_mass=np.array([0.0, 0.0, 0.8]),
        hand_point=np.array([0.45, 0.0, 1.0]),
        hand_force_on_robot=np.array([-55.0, 0.0, 0.0]),
        feet=feet(),
    )
    assert solve_contact_equilibrium(**common, friction_coefficient=0.8).feasible
    assert not solve_contact_equilibrium(**common, friction_coefficient=0.1).feasible


def test_tipping_moment_is_detected_even_with_high_friction():
    result = solve_contact_equilibrium(
        mass=50.0,
        center_of_mass=np.array([0.0, 0.0, 0.8]),
        hand_point=np.array([0.45, 0.0, 1.0]),
        hand_force_on_robot=np.array([-180.0, 0.0, 0.0]),
        feet=feet(),
        friction_coefficient=1.5,
    )
    assert not result.feasible
