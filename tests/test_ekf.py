import math

import numpy as np

from fsd.ekf import CtrvEkf

DT = 0.5


def _drive(truth, noise=0.0, seed=0):
    """Filter fed (x, y, yaw) measurements of `truth` (N, 3), one every DT; returns the filter."""
    rng = np.random.default_rng(seed)
    ekf = CtrvEkf(*truth[0])
    for z in truth[1:]:
        ekf.predict(DT)
        ekf.update(z + rng.normal(0.0, noise, 3) * [1.0, 1.0, 0.05])
    return ekf


def _straight(speed=8.0, yaw=0.3, n=20):
    t = DT * np.arange(n)
    return np.stack([speed * t * math.cos(yaw), speed * t * math.sin(yaw), np.full(n, yaw)], axis=1)


def _turn(speed=6.0, rate=0.25, n=20):
    t = DT * np.arange(n)
    yaw = rate * t
    return np.stack([speed / rate * np.sin(yaw), speed / rate * (1 - np.cos(yaw)), yaw], axis=1)


def test_filtered_speed_beats_a_frame_to_frame_difference():
    # 0.3 m of position noise: what the old tracker's finite difference turned into ~0.7 m/s of speed noise.
    truth, filtered, differenced = _straight(), [], []
    for seed in range(50):
        ekf = _drive(truth, noise=0.3, seed=seed)
        filtered.append(abs(ekf.x[2] - 8.0))
        assert abs(math.degrees(ekf.x[3] - 0.3)) < 5.0
        rng = np.random.default_rng(1000 + seed)
        a, b = (truth[i, :2] + rng.normal(0.0, 0.3, 2) for i in (-2, -1))
        differenced.append(abs(np.linalg.norm(b - a) / DT - 8.0))
    assert np.mean(filtered) < 0.5 * np.mean(differenced) and np.mean(filtered) < 0.4


def test_straight_prediction_without_turn_rate():
    ekf = CtrvEkf(0.0, 0.0, 0.0)
    ekf.x[2] = 10.0
    ekf.predict(1.0)
    assert np.allclose(ekf.x[:2], [10.0, 0.0])


def test_curved_forecast_beats_a_straight_line_on_a_turn():
    truth = _turn(n=30)
    ekf = _drive(truth[:20], noise=0.1)
    future = ekf.forecast((1.0, 2.0, 3.0))
    straight = truth[19, :2] + ekf.x[2] * np.array([math.cos(ekf.x[3]), math.sin(ekf.x[3])]) * np.array([[1.0], [2.0], [3.0]])
    target = truth[[21, 23, 25], :2]
    assert np.linalg.norm(future[-1] - target[-1]) < 0.5 * np.linalg.norm(straight[-1] - target[-1])


def test_heading_residual_wraps_across_pi():
    ekf = CtrvEkf(0.0, 0.0, math.radians(179.0))
    ekf.predict(DT)
    ekf.update(np.array([0.0, 0.0, math.radians(-179.0)]))
    assert abs(abs(ekf.x[3]) - math.pi) < math.radians(3.0)


def test_forecast_leaves_the_state_alone():
    ekf = _drive(_turn())
    before = ekf.x.copy()
    ekf.forecast((1.0, 3.0))
    assert np.array_equal(ekf.x, before)
