import numpy as np

from math_model.horizon import difference_residuals, joint_limit_barrier, joint_limit_margin, manipulability_metrics, minimum_jerk


def test_minimum_jerk_endpoint_conditions():
    eps = 1e-5
    assert minimum_jerk(0.0) == 0.0
    assert minimum_jerk(1.0) == 1.0
    assert abs((minimum_jerk(eps) - minimum_jerk(0.0)) / eps) < 1e-6
    assert abs((minimum_jerk(1.0) - minimum_jerk(1.0 - eps)) / eps) < 1e-6


def test_difference_residuals_vanish_for_constant_state():
    states = np.ones((8, 4))
    residual = difference_residuals(states, 0.1, velocity_weight=1.0, acceleration_weight=1.0, jerk_weight=1.0)
    assert np.allclose(residual, 0.0)


def test_joint_limit_barrier_activates_near_limit():
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 1.0])
    center = np.array([0.0, 0.0])
    near = np.array([0.95, -0.95])
    assert np.allclose(joint_limit_margin(center, lower, upper), 0.5)
    assert np.all(joint_limit_barrier(near, lower, upper) > joint_limit_barrier(center, lower, upper))


def test_manipulability_detects_singular_jacobian():
    good = manipulability_metrics(np.eye(3))
    bad = manipulability_metrics(np.diag([1.0, 1.0, 1e-5]))
    assert good.sigma_min == 1.0
    assert bad.sigma_min < good.sigma_min
    assert bad.condition_number > good.condition_number
