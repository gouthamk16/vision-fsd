# CTRV EKF Object Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the finite-difference + constant-velocity object tracker with a per-track Constant Turn-Rate & Velocity (CTRV) Extended Kalman Filter that denoises velocity, predicts curved paths, coasts through missed detections, and exposes position uncertainty.

**Architecture:** A standalone `CtrvEkf` (state `[px, py, v, yaw, yaw_rate]`, global frame) does pure filter math. `fsd/tracking.py` is refactored so a `Track` wraps one `CtrvEkf`; the two existing trackers become lifecycle managers (predict-all → associate → update/coast → spawn/retire) over a dict of `Track`. Filtered velocity, a curved CTRV forecast, and a position covariance are read out into `TrackedObject` and rendered in the BEV.

**Tech Stack:** Python 3.12, NumPy, pytest, OpenCV (viz only). Spec: `docs/superpowers/specs/2026-06-09-ctrv-ekf-tracking-design.md`.

---

## File Structure

- **Create `fsd/ekf.py`** — `CtrvEkf` class. Pure NumPy filter math (predict/update/forecast), no nuScenes deps.
- **Create `tests/test_ekf.py`** — filter math: Jacobian correctness, convergence, curved forecast.
- **Modify `fsd/tracking.py`** — replace finite-difference internals with `Track` + `CtrvEkf`; keep the two tracker classes and their public `update(frame, boxes)` / `reset()` API.
- **Rewrite `tests/test_tracking.py`** — association, predicted-position gating, coasting/retirement, ego-frame rotation, convergence (not exact-match).
- **Modify `fsd/world_model.py`** — add `position_cov_ego` passthrough on `WorldObject`; draw curved ghost + 1σ ellipse in `_draw_object_motion`.

Conventions to follow (existing code):
- nuScenes quaternion math via `fsd.lidar_projection.quaternion_to_rotation_matrix`, `transform_points`, `inverse_transform_points` (row-vector `Nx3`).
- Ego yaw from a rotation matrix `R`: `np.arctan2(R[1, 0], R[0, 0])`.
- Object global yaw = `ego_yaw + box.yaw_ego`.
- Synthesize boxes with `fsd.object_detection.make_ego_box(...)`.

---

## Task 1: CtrvEkf — construction and predict step

**Files:**
- Create: `fsd/ekf.py`
- Test: `tests/test_ekf.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ekf.py
import numpy as np

from fsd.ekf import CtrvEkf


def _ekf(px, py, v, yaw, w):
    x0 = np.array([px, py, v, yaw, w], dtype=np.float64)
    P0 = np.eye(5)
    return CtrvEkf(x0, P0, sigma_a=2.0, sigma_yaw_accel=0.5)


def test_predict_straight_line_when_yaw_rate_zero():
    ekf = _ekf(0.0, 0.0, 10.0, 0.0, 0.0)  # heading +x, 10 m/s
    ekf.predict(0.5)
    assert np.allclose(ekf.x[:2], [5.0, 0.0], atol=1e-9)
    assert ekf.x[2] == 10.0
    assert ekf.x[3] == 0.0


def test_predict_curves_with_yaw_rate():
    # v=pi m/s, w=pi/2 rad/s: quarter turn in 1 s, radius r=v/w=2.
    ekf = _ekf(0.0, 0.0, np.pi, 0.0, np.pi / 2)
    ekf.predict(1.0)
    # closed-form CTRV end point of a quarter circle starting heading +x, center (0, r)
    assert np.allclose(ekf.x[:2], [2.0, 2.0], atol=1e-9)
    assert np.allclose(ekf.x[3], np.pi / 2, atol=1e-9)


def test_predict_grows_covariance():
    ekf = _ekf(0.0, 0.0, 10.0, 0.0, 0.0)
    before = np.trace(ekf.P)
    ekf.predict(0.5)
    assert np.trace(ekf.P) > before
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ekf.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fsd.ekf'`

- [ ] **Step 3: Write minimal implementation**

```python
# fsd/ekf.py
from __future__ import annotations

import numpy as np

_EPS = 1e-4  # below this |yaw_rate| we use the straight-line limit


def wrap_angle(a: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class CtrvEkf:
    """Extended Kalman filter on a CTRV model.

    State x = [px, py, v, yaw, yaw_rate] in a fixed (global) frame.
    Nonlinearity is in the motion model only; the measurement model is linear.
    """

    def __init__(self, x0: np.ndarray, P0: np.ndarray, sigma_a: float, sigma_yaw_accel: float):
        self.x = np.asarray(x0, dtype=np.float64).copy()
        self.P = np.asarray(P0, dtype=np.float64).copy()
        self.sigma_a = float(sigma_a)
        self.sigma_yaw_accel = float(sigma_yaw_accel)

    def predict(self, dt: float) -> None:
        px, py, v, yaw, w = self.x
        F = np.eye(5)
        if abs(w) > _EPS:
            s0, c0 = np.sin(yaw), np.cos(yaw)
            s1, c1 = np.sin(yaw + w * dt), np.cos(yaw + w * dt)
            px += v / w * (s1 - s0)
            py += v / w * (c0 - c1)
            yaw = yaw + w * dt
            F[0, 2] = (s1 - s0) / w
            F[0, 3] = v / w * (c1 - c0)
            F[0, 4] = v * dt / w * c1 - v / w**2 * (s1 - s0)
            F[1, 2] = (c0 - c1) / w
            F[1, 3] = v / w * (s1 - s0)
            F[1, 4] = v * dt / w * s1 - v / w**2 * (c0 - c1)
        else:
            s0, c0 = np.sin(yaw), np.cos(yaw)
            px += v * c0 * dt
            py += v * s0 * dt
            yaw = yaw + w * dt
            F[0, 2] = c0 * dt
            F[0, 3] = -v * s0 * dt
            F[0, 4] = -0.5 * v * dt**2 * s0
            F[1, 2] = s0 * dt
            F[1, 3] = v * c0 * dt
            F[1, 4] = 0.5 * v * dt**2 * c0
        F[3, 4] = dt

        self.x = np.array([px, py, v, wrap_angle(yaw), w])
        self.P = F @ self.P @ F.T + self._process_noise(dt, self.x[3])

    def _process_noise(self, dt: float, yaw: float) -> np.ndarray:
        c, s = np.cos(yaw), np.sin(yaw)
        G = np.array([
            [0.5 * dt**2 * c, 0.0],
            [0.5 * dt**2 * s, 0.0],
            [dt, 0.0],
            [0.0, 0.5 * dt**2],
            [0.0, dt],
        ])
        Q_nu = np.diag([self.sigma_a**2, self.sigma_yaw_accel**2])
        return G @ Q_nu @ G.T
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ekf.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add fsd/ekf.py tests/test_ekf.py
git commit -m "feat: CtrvEkf predict step + process noise"
```

---

## Task 2: CtrvEkf — Jacobian sanity via numerical differentiation

Guards the hand-derived `F` against algebra errors. No new production code if `F` is correct.

**Files:**
- Test: `tests/test_ekf.py`

- [ ] **Step 1: Write the failing test**

```python
def test_predict_jacobian_matches_numerical():
    # Compare analytic F (captured from predict) against a finite-difference Jacobian
    # of the deterministic state transition.
    def transition(state, dt):
        e = CtrvEkf(state, np.eye(5), sigma_a=0.0, sigma_yaw_accel=0.0)
        e.predict(dt)
        return e.x.copy()

    x = np.array([3.0, -1.0, 8.0, 0.6, 0.3])
    dt = 0.4
    n = len(x)
    num = np.zeros((n, n))
    eps = 1e-6
    for j in range(n):
        dx = np.zeros(n); dx[j] = eps
        num[:, j] = (transition(x + dx, dt) - transition(x - dx, dt)) / (2 * eps)

    ekf = CtrvEkf(x, np.eye(5), sigma_a=0.0, sigma_yaw_accel=0.0)
    F = ekf.analytic_F(dt)
    assert np.allclose(F, num, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ekf.py::test_predict_jacobian_matches_numerical -v`
Expected: FAIL — `AttributeError: 'CtrvEkf' object has no attribute 'analytic_F'`

- [ ] **Step 3: Write minimal implementation**

Refactor `predict` to build `F` via a shared helper, exposed for testing. Replace the `F` construction inside `predict` with a call to `analytic_F`, then use it:

```python
    def analytic_F(self, dt: float) -> np.ndarray:
        px, py, v, yaw, w = self.x
        F = np.eye(5)
        if abs(w) > _EPS:
            s0, c0 = np.sin(yaw), np.cos(yaw)
            s1, c1 = np.sin(yaw + w * dt), np.cos(yaw + w * dt)
            F[0, 2] = (s1 - s0) / w
            F[0, 3] = v / w * (c1 - c0)
            F[0, 4] = v * dt / w * c1 - v / w**2 * (s1 - s0)
            F[1, 2] = (c0 - c1) / w
            F[1, 3] = v / w * (s1 - s0)
            F[1, 4] = v * dt / w * s1 - v / w**2 * (c0 - c1)
        else:
            s0, c0 = np.sin(yaw), np.cos(yaw)
            F[0, 2] = c0 * dt
            F[0, 3] = -v * s0 * dt
            F[0, 4] = -0.5 * v * dt**2 * s0
            F[1, 2] = s0 * dt
            F[1, 3] = v * c0 * dt
            F[1, 4] = 0.5 * v * dt**2 * c0
        F[3, 4] = dt
        return F
```

Then in `predict`, replace the inline `F` block with `F = self.analytic_F(dt)` computed **before** mutating `self.x`, and propagate the mean using the same closed form:

```python
    def predict(self, dt: float) -> None:
        F = self.analytic_F(dt)
        px, py, v, yaw, w = self.x
        if abs(w) > _EPS:
            s0, c0 = np.sin(yaw), np.cos(yaw)
            s1, c1 = np.sin(yaw + w * dt), np.cos(yaw + w * dt)
            px += v / w * (s1 - s0)
            py += v / w * (c0 - c1)
        else:
            px += v * np.cos(yaw) * dt
            py += v * np.sin(yaw) * dt
        yaw = yaw + w * dt
        self.x = np.array([px, py, v, wrap_angle(yaw), w])
        self.P = F @ self.P @ F.T + self._process_noise(dt, self.x[3])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ekf.py -v`
Expected: PASS (4 tests — predict tests still pass after refactor)

- [ ] **Step 5: Commit**

```bash
git add fsd/ekf.py tests/test_ekf.py
git commit -m "test: numerical-Jacobian guard for CtrvEkf predict"
```

---

## Task 3: CtrvEkf — measurement update

**Files:**
- Modify: `fsd/ekf.py`
- Test: `tests/test_ekf.py`

- [ ] **Step 1: Write the failing test**

```python
def test_update_pulls_state_toward_measurement():
    ekf = _ekf(0.0, 0.0, 0.0, 0.0, 0.0)
    ekf.P = np.diag([1.0, 1.0, 100.0, 1.0, 100.0])  # uncertain v, yaw_rate
    ekf.update(np.array([1.0, 0.5, 0.3]), sigma_pos=0.5, sigma_yaw=0.1)
    assert 0.0 < ekf.x[0] <= 1.0
    assert 0.0 < ekf.x[1] <= 0.5
    assert abs(wrap := ekf.x[3]) <= 0.3 + 1e-9 and wrap > 0.0


def test_update_wraps_yaw_innovation():
    # Measurement just below +pi vs state just above -pi: true error is tiny, not ~2pi.
    ekf = _ekf(0.0, 0.0, 5.0, -np.pi + 0.05, 0.0)
    ekf.P = np.diag([0.1, 0.1, 1.0, 1.0, 1.0])
    ekf.update(np.array([0.0, 0.0, np.pi - 0.05]), sigma_pos=0.5, sigma_yaw=0.1)
    # yaw should stay near the -pi/+pi seam, not jump toward 0
    assert abs(wrap_angle(ekf.x[3])) > np.pi - 0.2


def test_update_shrinks_covariance():
    ekf = _ekf(0.0, 0.0, 0.0, 0.0, 0.0)
    before = np.trace(ekf.P)
    ekf.update(np.array([0.2, 0.0, 0.0]), sigma_pos=0.5, sigma_yaw=0.1)
    assert np.trace(ekf.P) < before
```

Add the import at the top of the test file if not present: `from fsd.ekf import CtrvEkf, wrap_angle`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ekf.py -k update -v`
Expected: FAIL — `AttributeError: 'CtrvEkf' object has no attribute 'update'`

- [ ] **Step 3: Write minimal implementation**

```python
    _H = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
    ])

    def update(self, z: np.ndarray, sigma_pos: float, sigma_yaw: float) -> None:
        H = self._H
        R = np.diag([sigma_pos**2, sigma_pos**2, sigma_yaw**2])
        y = z - H @ self.x
        y[2] = wrap_angle(y[2])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[3] = wrap_angle(self.x[3])
        I = np.eye(5)
        self.P = (I - K @ H) @ self.P
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ekf.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add fsd/ekf.py tests/test_ekf.py
git commit -m "feat: CtrvEkf linear measurement update with yaw wrapping"
```

---

## Task 4: CtrvEkf — curved forecast + convergence on a turn

**Files:**
- Modify: `fsd/ekf.py`
- Test: `tests/test_ekf.py`

- [ ] **Step 1: Write the failing test**

```python
def test_forecast_returns_points_on_the_turn():
    # Quarter-circle motion: forecast must curve, not go straight.
    ekf = _ekf(0.0, 0.0, np.pi, 0.0, np.pi / 2)  # r = 2
    fut = ekf.forecast((1.0, 2.0))
    assert fut.shape == (2, 2)
    assert np.allclose(fut[0], [2.0, 2.0], atol=1e-9)   # quarter turn
    assert np.allclose(fut[1], [0.0, 4.0], atol=1e-9)   # half turn
    # a straight constant-velocity guess at t=1s would be [pi, 0]; confirm we are not that
    assert not np.allclose(fut[0], [np.pi, 0.0], atol=0.1)


def test_filter_converges_on_noisy_turn():
    rng = np.random.default_rng(0)
    dt = 0.1
    true = np.array([0.0, 0.0, 8.0, 0.0, 0.4])  # constant turn
    truth = CtrvEkf(true.copy(), np.eye(5), 0.0, 0.0)
    ekf = CtrvEkf(
        np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        np.diag([0.5, 0.5, 100.0, 0.5, 100.0]),
        sigma_a=2.0, sigma_yaw_accel=0.5,
    )
    for _ in range(80):
        truth.predict(dt)
        z = truth.x[[0, 1, 3]] + rng.normal(0, [0.5, 0.5, 0.1])
        ekf.predict(dt)
        ekf.update(z, sigma_pos=0.5, sigma_yaw=0.1)
    assert abs(ekf.x[2] - 8.0) < 1.0      # speed within 1 m/s
    assert abs(ekf.x[4] - 0.4) < 0.15     # yaw-rate within 0.15 rad/s
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ekf.py -k "forecast or converges" -v`
Expected: FAIL — `AttributeError: 'CtrvEkf' object has no attribute 'forecast'`

- [ ] **Step 3: Write minimal implementation**

```python
    def forecast(self, horizons_s: tuple[float, ...]) -> np.ndarray:
        """Closed-form CTRV positions at each horizon (global frame). Returns (K, 2)."""
        px, py, v, yaw, w = self.x
        out = []
        for h in horizons_s:
            if abs(w) > _EPS:
                fx = px + v / w * (np.sin(yaw + w * h) - np.sin(yaw))
                fy = py + v / w * (np.cos(yaw) - np.cos(yaw + w * h))
            else:
                fx = px + v * np.cos(yaw) * h
                fy = py + v * np.sin(yaw) * h
            out.append([fx, fy])
        return np.array(out, dtype=np.float64)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ekf.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add fsd/ekf.py tests/test_ekf.py
git commit -m "feat: CtrvEkf closed-form curved forecast"
```

---

## Task 5: Track wrapper + measurement helper in tracking.py

Introduce the per-track filter wrapper and the global-measurement extraction, plus filter tuning constants. The two tracker classes are rewired in Tasks 6–7; this task adds the shared pieces and keeps the file importable.

**Files:**
- Modify: `fsd/tracking.py` (top-of-file helpers + new `Track`; leave the two tracker classes for the next tasks)
- Test: `tests/test_tracking.py` (add a focused test; full rewrite happens in Task 7)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tracking.py` (existing imports stay for now):

```python
from fsd.tracking import Track, measurement_global


def test_measurement_global_adds_ego_yaw():
    # ego yaw +90deg; a box with yaw_ego=0 reads as global yaw +pi/2.
    from fsd.tracking import measurement_global
    yaw90 = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
    frame = _frame([0.0, 0.0, 0.0], yaw90, 0)
    box = _box([10.0, 0.0, 0.0], "obj")  # yaw_ego = 0
    z = measurement_global(box, frame.ego_pose)
    assert np.allclose(z[:2], [0.0, 10.0], atol=1e-6)  # +x ego -> +y global under +90deg
    assert abs(z[2] - np.pi / 2) < 1e-6


def test_track_initializes_at_measurement():
    z = np.array([3.0, 4.0, 0.2])
    t = Track("id0", "car", z, ts=0, size=np.array([1.9, 4.6, 1.7]))
    assert np.allclose(t.ekf.x[:2], [3.0, 4.0])
    assert t.ekf.x[2] == 0.0 and t.ekf.x[4] == 0.0
    assert abs(t.ekf.x[3] - 0.2) < 1e-12
    assert t.misses == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tracking.py -k "measurement_global or Track" -v`
Expected: FAIL — `ImportError: cannot import name 'Track'`

- [ ] **Step 3: Write minimal implementation**

At the top of `fsd/tracking.py`, replace the module docstring's stale "finite difference" description and add (keep `from __future__ import annotations`):

```python
import numpy as np

from fsd.data import SurroundFrame
from fsd.ekf import CtrvEkf
from fsd.lidar_projection import (
    inverse_transform_points,
    quaternion_to_rotation_matrix,
    transform_points,
)
from fsd.object_detection import Box3D, make_ego_box

# Filter tuning (see spec).
SIGMA_POS = 0.5
SIGMA_YAW = 0.1
SIGMA_A = 2.0
SIGMA_YAW_ACCEL = 0.5
INIT_VAR_V = 100.0       # large prior on speed
INIT_VAR_YAW_RATE = 100.0


def measurement_global(box: Box3D, ego_pose: dict) -> np.ndarray:
    """Detection as [px, py, yaw] in the global frame."""
    g = transform_points(box.center_ego.reshape(1, 3), ego_pose["rotation"], ego_pose["translation"])[0, :2]
    rot = quaternion_to_rotation_matrix(ego_pose["rotation"])
    ego_yaw = np.arctan2(rot[1, 0], rot[0, 0])
    return np.array([g[0], g[1], ego_yaw + box.yaw_ego])


class Track:
    """One CTRV filter plus identity, class, last-seen size and miss count."""

    def __init__(self, track_id: str, class_name: str, z: np.ndarray, ts: int, size: np.ndarray):
        x0 = np.array([z[0], z[1], 0.0, z[2], 0.0])
        P0 = np.diag([SIGMA_POS**2, SIGMA_POS**2, INIT_VAR_V, SIGMA_YAW**2, INIT_VAR_YAW_RATE])
        self.ekf = CtrvEkf(x0, P0, sigma_a=SIGMA_A, sigma_yaw_accel=SIGMA_YAW_ACCEL)
        self.id = track_id
        self.class_name = class_name
        self.size = np.asarray(size, dtype=np.float64).copy()
        self.misses = 0

    def predict(self, dt: float) -> None:
        if dt > 0.0:
            self.ekf.predict(dt)

    def correct(self, z: np.ndarray, size: np.ndarray) -> None:
        self.ekf.update(z, sigma_pos=SIGMA_POS, sigma_yaw=SIGMA_YAW)
        self.size = np.asarray(size, dtype=np.float64).copy()
        self.misses = 0

    @property
    def global_xy(self) -> np.ndarray:
        return self.ekf.x[:2]
```

Keep the existing `TrackedObject` dataclass and the two tracker classes in place for now (they will be rewritten next). The legacy `_global_xy` / `_make_tracked` functions can remain temporarily — they are removed in Task 7.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tracking.py -k "measurement_global or Track" -v`
Expected: PASS (2 new tests; legacy tests may still pass since old code is untouched)

- [ ] **Step 5: Commit**

```bash
git add fsd/tracking.py tests/test_tracking.py
git commit -m "feat: Track wrapper + global measurement helper for CTRV tracking"
```

---

## Task 6: Read-out helper — Track to TrackedObject in ego frame

Converts filtered global state into the ego-frame `TrackedObject` (filtered velocity, curved forecast, position covariance). Shared by both trackers.

**Files:**
- Modify: `fsd/tracking.py`
- Test: `tests/test_tracking.py`

- [ ] **Step 1: Write the failing test**

```python
from fsd.tracking import HORIZONS_S, track_to_object


def test_track_to_object_reports_filtered_velocity_in_ego():
    identity = [1.0, 0.0, 0.0, 0.0]
    frame = _frame([0.0, 0.0, 0.0], identity, 0)
    z = measurement_global(_box([20.0, 0.0, 0.0], "obj"), frame.ego_pose)
    t = Track("id0", "car", z, ts=0, size=np.array([1.9, 4.6, 1.7]))
    # inject a clean eastbound state: 6 m/s along global +x
    t.ekf.x = np.array([20.0, 0.0, 6.0, 0.0, 0.0])
    box = _box([20.0, 0.0, 0.0], "obj")
    obj = track_to_object(t, box, frame.ego_pose)
    assert abs(obj.speed_mps - 6.0) < 1e-6
    assert np.allclose(obj.velocity_ego, [6.0, 0.0], atol=1e-6)
    assert obj.future_xy_ego.shape == (len(HORIZONS_S), 2)
    assert np.allclose(obj.future_xy_ego[0], [26.0, 0.0], atol=1e-6)  # 1 s
    assert obj.position_cov_ego.shape == (2, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tracking.py -k track_to_object -v`
Expected: FAIL — `ImportError: cannot import name 'track_to_object'`

- [ ] **Step 3: Write minimal implementation**

First extend `TrackedObject` to carry the covariance. Replace the existing dataclass with:

```python
@dataclass(frozen=True)
class TrackedObject:
    box: Box3D
    track_id: str
    velocity_ego: np.ndarray   # (vx, vy) m/s, current ego frame
    speed_mps: float
    future_xy_ego: np.ndarray  # (K, 2) curved CTRV forecast, current ego frame
    position_cov_ego: np.ndarray  # (2, 2) position covariance, current ego frame
```

Add the horizon constant near the other tuning constants:

```python
HORIZONS_S = (1.0, 2.0, 3.0)
MIN_SPEED_MPS = 0.5
```

Then add the read-out helper:

```python
def _global_to_ego_xy(points_xy: np.ndarray, ego_pose: dict) -> np.ndarray:
    pts3 = np.column_stack([points_xy, np.zeros(len(points_xy))])
    ego = inverse_transform_points(pts3, ego_pose["rotation"], ego_pose["translation"])
    return ego[:, :2]


def track_to_object(track: Track, box: Box3D, ego_pose: dict) -> TrackedObject:
    rot = quaternion_to_rotation_matrix(ego_pose["rotation"])[:2, :2]
    v, yaw = track.ekf.x[2], track.ekf.x[3]
    v_global = np.array([v * np.cos(yaw), v * np.sin(yaw)])
    v_ego = rot.T @ v_global
    speed = float(abs(v))
    if speed >= MIN_SPEED_MPS:
        future_ego = _global_to_ego_xy(track.ekf.forecast(HORIZONS_S), ego_pose)
    else:
        future_ego = np.empty((0, 2))
    cov_ego = rot.T @ track.ekf.P[:2, :2] @ rot
    return TrackedObject(
        box=box,
        track_id=track.id,
        velocity_ego=v_ego,
        speed_mps=speed,
        future_xy_ego=future_ego,
        position_cov_ego=cov_ego,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tracking.py -k track_to_object -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add fsd/tracking.py tests/test_tracking.py
git commit -m "feat: ego-frame read-out from CTRV track (velocity, curved forecast, covariance)"
```

---

## Task 7: Rewire both trackers (predict → associate → update/coast → spawn/retire)

Replaces the finite-difference bodies of `GtVelocityTracker` and `PredictionVelocityTracker`, deletes `_global_xy` / `_make_tracked`, and rewrites the legacy tests to convergence/lifecycle assertions.

**Files:**
- Modify: `fsd/tracking.py`
- Rewrite: `tests/test_tracking.py`

- [ ] **Step 1: Write the failing tests** (replace the whole file)

```python
# tests/test_tracking.py
from pathlib import Path

import numpy as np

from fsd.data import CameraFrame, SurroundFrame
from fsd.object_detection import Box3D, box_bottom_corners_ego
from fsd.tracking import (
    HORIZONS_S,
    Track,
    GtVelocityTracker,
    PredictionVelocityTracker,
    measurement_global,
    track_to_object,
)

IDENTITY = [1.0, 0.0, 0.0, 0.0]


def _frame(ego_translation, ego_rotation, timestamp_us):
    pose = {"rotation": list(ego_rotation), "translation": list(ego_translation)}
    cam = CameraFrame("CAM_FRONT", Path("x.jpg"), timestamp_us, "sd", {}, pose)
    return SurroundFrame("scene", "scene-x", "sample", 0, timestamp_us, {"CAM_FRONT": cam})


def _box(center_ego, token, class_name="car"):
    center_ego = np.asarray(center_ego, dtype=np.float64)
    size = np.array([1.9, 4.6, 1.7])
    return Box3D(
        sample_token="s", annotation_token="a", class_name=class_name,
        raw_category="vehicle.car", center_ego=center_ego, size=size, yaw_ego=0.0,
        corners_ego=box_bottom_corners_ego(center_ego, size, 0.0),
        num_lidar_pts=10, num_radar_pts=0, instance_token=token,
    )


def _drive(tracker, positions, dt_us=500_000):
    """Feed one straight-moving box through the tracker; return final TrackedObject."""
    out = None
    for k, x in enumerate(positions):
        out = tracker.update(_frame([0.0, 0.0, 0.0], IDENTITY, k * dt_us), [_box([x, 0.0, 0.0], "obj")])
    return out[0]


# --- helpers already covered in Tasks 5-6 ---

def test_measurement_global_adds_ego_yaw():
    yaw90 = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
    z = measurement_global(_box([10.0, 0.0, 0.0], "obj"), _frame([0, 0, 0], yaw90, 0).ego_pose)
    assert np.allclose(z[:2], [0.0, 10.0], atol=1e-6)
    assert abs(z[2] - np.pi / 2) < 1e-6


def test_track_initializes_at_measurement():
    z = np.array([3.0, 4.0, 0.2])
    t = Track("id0", "car", z, ts=0, size=np.array([1.9, 4.6, 1.7]))
    assert np.allclose(t.ekf.x[:2], [3.0, 4.0]) and t.misses == 0


# --- GT tracker (token association) ---

def test_first_sighting_has_zero_velocity():
    tracker = GtVelocityTracker()
    out = tracker.update(_frame([0, 0, 0], IDENTITY, 0), [_box([5.0, 0.0, 0.0], "obj")])
    assert out[0].speed_mps == 0.0
    assert out[0].future_xy_ego.shape == (0, 2)


def test_filtered_speed_converges_to_truth():
    # box moves +5 m/s along ego +x (2.5 m per 0.5 s); filter should converge near 5.
    tracker = GtVelocityTracker()
    positions = [5.0 + 2.5 * k for k in range(20)]
    obj = _drive(tracker, positions)
    assert abs(obj.speed_mps - 5.0) < 0.5
    assert np.allclose(obj.velocity_ego, [5.0, 0.0], atol=0.6)
    assert obj.future_xy_ego.shape == (len(HORIZONS_S), 2)


def test_reset_clears_history():
    tracker = GtVelocityTracker()
    tracker.update(_frame([0, 0, 0], IDENTITY, 0), [_box([20.0, 0.0, 0.0], "obj")])
    tracker.reset()
    out = tracker.update(_frame([0, 0, 0], IDENTITY, 500_000), [_box([22.5, 0.0, 0.0], "obj")])
    assert out[0].speed_mps == 0.0


def test_velocity_rotates_into_ego_frame():
    # ego yaw +90deg; a global +x mover reads as ego -y motion after convergence.
    tracker = GtVelocityTracker()
    yaw90 = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
    obj = None
    for k in range(20):
        # global +x motion: under +90deg ego, that is ego -y; box center ego_y decreases.
        out = tracker.update(_frame([0, 0, 0], yaw90, k * 500_000), [_box([10.0, -2.5 * k, 0.0], "obj")])
        obj = out[0]
    assert obj.velocity_ego[1] < -3.0 and abs(obj.velocity_ego[0]) < 1.5


# --- Prediction tracker (greedy nearest-by-class, coasting) ---

def test_prediction_association_recovers_two_tracks():
    tracker = PredictionVelocityTracker()
    out = None
    for k in range(20):
        out = tracker.update(
            _frame([0, 0, 0], IDENTITY, k * 500_000),
            [_box([20.0 + 2.5 * k, 2.0, 0.0], ""), _box([10.0 + 0.5 * k, -3.0, 0.0], "")],
        )
    speeds = sorted(o.speed_mps for o in out)
    assert abs(speeds[0] - 1.0) < 0.5    # 0.5 m / 0.5 s
    assert abs(speeds[1] - 5.0) < 0.6    # 2.5 m / 0.5 s


def test_prediction_gate_rejects_teleport():
    tracker = PredictionVelocityTracker(gate_m=5.0)
    tracker.update(_frame([0, 0, 0], IDENTITY, 0), [_box([20.0, 0.0, 0.0], "")])
    out = tracker.update(_frame([0, 0, 0], IDENTITY, 500_000), [_box([40.0, 0.0, 0.0], "")])
    assert out[0].speed_mps == 0.0  # 20 m jump > gate -> fresh track, no velocity


def test_prediction_class_must_match():
    tracker = PredictionVelocityTracker()
    tracker.update(_frame([0, 0, 0], IDENTITY, 0), [_box([20.0, 0.0, 0.0], "", class_name="car")])
    out = tracker.update(_frame([0, 0, 0], IDENTITY, 500_000), [_box([21.0, 0.0, 0.0], "", class_name="truck")])
    assert out[0].speed_mps == 0.0


def test_prediction_coasts_through_one_missed_frame():
    # Object seen, then missing one frame (< max_age), then seen again: same track id survives.
    tracker = PredictionVelocityTracker(max_age_s=0.6)
    tracker.update(_frame([0, 0, 0], IDENTITY, 0), [_box([20.0, 0.0, 0.0], "")])
    out1 = tracker.update(_frame([0, 0, 0], IDENTITY, 500_000), [_box([22.5, 0.0, 0.0], "")])
    tid = out1[0].track_id
    coasted = tracker.update(_frame([0, 0, 0], IDENTITY, 1_000_000), [])  # missed frame
    assert any(o.track_id == tid for o in coasted)  # still alive, coasted forward


def test_prediction_retires_after_max_age():
    tracker = PredictionVelocityTracker(max_age_s=0.6)
    tracker.update(_frame([0, 0, 0], IDENTITY, 0), [_box([20.0, 0.0, 0.0], "")])
    tracker.update(_frame([0, 0, 0], IDENTITY, 500_000), [])     # miss 1 (0.5 s)
    out = tracker.update(_frame([0, 0, 0], IDENTITY, 1_000_000), [])  # miss 2 (1.0 s > 0.6)
    assert out == []  # track dropped
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tracking.py -v`
Expected: FAIL — `GtVelocityTracker.update` still returns old finite-difference objects (missing `position_cov_ego`, no coasting, no `max_age_s` kwarg).

- [ ] **Step 3: Write the implementation** (replace both tracker classes; delete `_global_xy` and `_make_tracked`)

```python
class GtVelocityTracker:
    """CTRV filter per object, associated by stable nuScenes instance_token (the oracle)."""

    def __init__(self, max_age_s: float = 0.5):
        self.max_age_s = max_age_s
        self._tracks: dict[str, Track] = {}
        self._prev_ts: int | None = None

    def reset(self) -> None:
        self._tracks = {}
        self._prev_ts = None

    def update(self, frame: SurroundFrame, boxes: list[Box3D]) -> list[TrackedObject]:
        ego_pose = frame.ego_pose
        ts = frame.timestamp_us
        dt = 0.0 if self._prev_ts is None else (ts - self._prev_ts) / 1_000_000.0
        self._prev_ts = ts
        for track in self._tracks.values():
            track.predict(dt)
            track.misses += 1  # cleared on correct()

        objects: list[TrackedObject] = []
        seen: set[str] = set()
        for box in boxes:
            token = box.instance_token
            if not token:
                continue  # GT oracle requires identity
            z = measurement_global(box, ego_pose)
            track = self._tracks.get(token)
            if track is None or track.class_name != box.class_name:
                track = Track(token, box.class_name, z, ts, box.size)
                self._tracks[token] = track
            else:
                track.correct(z, box.size)
            seen.add(token)
            objects.append(track_to_object(track, box, ego_pose))

        objects += self._coast_unseen(seen, dt, ego_pose)
        self._retire(dt)
        return objects

    def _coast_unseen(self, seen, dt, ego_pose):
        out = []
        for tid, track in self._tracks.items():
            if tid in seen:
                continue
            box = _coasted_box(track, ego_pose)
            out.append(track_to_object(track, box, ego_pose))
        return out

    def _retire(self, dt: float) -> None:
        if dt <= 0.0:
            return
        max_misses = max(1, round(self.max_age_s / dt))
        self._tracks = {tid: t for tid, t in self._tracks.items() if t.misses <= max_misses}
```

```python
class PredictionVelocityTracker:
    """CTRV filter per object, greedy nearest-by-class association for identity-free boxes."""

    def __init__(self, max_age_s: float = 0.5, gate_m: float = 5.0):
        self.max_age_s = max_age_s
        self.gate_m = gate_m
        self._tracks: dict[str, Track] = {}
        self._prev_ts: int | None = None
        self._next_id = 0

    def reset(self) -> None:
        self._tracks = {}
        self._prev_ts = None
        self._next_id = 0

    def update(self, frame: SurroundFrame, boxes: list[Box3D]) -> list[TrackedObject]:
        ego_pose = frame.ego_pose
        ts = frame.timestamp_us
        dt = 0.0 if self._prev_ts is None else (ts - self._prev_ts) / 1_000_000.0
        self._prev_ts = ts
        for track in self._tracks.values():
            track.predict(dt)
            track.misses += 1

        zs = [measurement_global(box, ego_pose) for box in boxes]
        # gate on the predicted position
        candidates = []
        for i, box in enumerate(boxes):
            for tid, track in self._tracks.items():
                if track.class_name != box.class_name:
                    continue
                dist = float(np.hypot(*(zs[i][:2] - track.global_xy)))
                if dist <= self.gate_m:
                    candidates.append((dist, i, tid))
        candidates.sort(key=lambda c: c[0])

        box_to_tid: dict[int, str] = {}
        used: set[str] = set()
        for _, i, tid in candidates:
            if i in box_to_tid or tid in used:
                continue
            box_to_tid[i] = tid
            used.add(tid)

        objects: list[TrackedObject] = []
        seen: set[str] = set()
        for i, box in enumerate(boxes):
            tid = box_to_tid.get(i)
            if tid is None:
                tid = f"p{self._next_id}"
                self._next_id += 1
                track = Track(tid, box.class_name, zs[i], ts, box.size)
                self._tracks[tid] = track
            else:
                track = self._tracks[tid]
                track.correct(zs[i], box.size)
            seen.add(tid)
            objects.append(track_to_object(track, box, ego_pose))

        for tid, track in self._tracks.items():
            if tid in seen:
                continue
            objects.append(track_to_object(track, _coasted_box(track, ego_pose), ego_pose))

        if dt > 0.0:
            max_misses = max(1, round(self.max_age_s / dt))
            self._tracks = {tid: t for tid, t in self._tracks.items() if t.misses <= max_misses}
        return objects
```

Add the coasted-box synthesizer (filtered global state → ego-frame `Box3D`):

```python
def _coasted_box(track: Track, ego_pose: dict) -> Box3D:
    px, py, _, yaw, _ = track.ekf.x
    center_ego = inverse_transform_points(
        np.array([[px, py, 0.0]]), ego_pose["rotation"], ego_pose["translation"]
    )[0]
    rot = quaternion_to_rotation_matrix(ego_pose["rotation"])
    ego_yaw = np.arctan2(rot[1, 0], rot[0, 0])
    return make_ego_box(
        sample_token="", class_name=track.class_name,
        center_ego=center_ego, size=track.size, yaw_ego=float(yaw - ego_yaw), source="coast",
    )
```

Delete the now-unused `_global_xy` and `_make_tracked` functions.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tracking.py tests/test_ekf.py -v`
Expected: PASS (all tracking + EKF tests)

- [ ] **Step 5: Commit**

```bash
git add fsd/tracking.py tests/test_tracking.py
git commit -m "feat: CTRV-EKF trackers with predicted-position gating and coasting"
```

---

## Task 8: World model passthrough + BEV visualization

Expose `position_cov_ego` on `WorldObject` and draw the curved ghost + 1σ ellipse.

**Files:**
- Modify: `fsd/world_model.py` (`WorldObject`, `_tracked_world_object`, `_draw_object_motion`)

- [ ] **Step 1: Write the failing test**

Create `tests/test_world_model_viz.py`:

```python
import numpy as np

from fsd.world_model import WorldObject, _draw_object_motion
from fsd.object_detection import make_ego_box


def _obj(speed, cov):
    box = make_ego_box("", "car", np.array([20.0, 0.0, 0.0]), np.array([1.9, 4.6, 1.7]), 0.0)
    return WorldObject(
        box=box,
        distance_m=20.0,
        footprint_ego=box.corners_ego[:, :2].copy(),
        track_id="t0",
        velocity_ego=np.array([speed, 0.0]),
        speed_mps=speed,
        future_xy_ego=np.array([[26.0, 1.0], [32.0, 4.0], [38.0, 9.0]]),  # curved
        position_cov_ego=cov,
    )


def test_draw_object_motion_counts_moving_and_runs_with_cov():
    img = np.zeros((900, 400, 3), dtype=np.uint8)
    objs = [_obj(5.0, np.diag([1.0, 4.0]))]
    moving = _draw_object_motion(img, objs, (-50.0, 50.0), (-50.0, 50.0), 0.25, scale=2)
    assert moving == 1
    assert img.any()  # something was drawn


def test_draw_object_motion_skips_slow():
    img = np.zeros((900, 400, 3), dtype=np.uint8)
    objs = [_obj(0.1, np.diag([1.0, 1.0]))]
    assert _draw_object_motion(img, objs, (-50.0, 50.0), (-50.0, 50.0), 0.25, scale=2) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_world_model_viz.py -v`
Expected: FAIL — `TypeError: WorldObject.__init__() got an unexpected keyword argument 'position_cov_ego'`

- [ ] **Step 3: Write minimal implementation**

In `fsd/world_model.py`, add the field to `WorldObject`:

```python
@dataclass(frozen=True)
class WorldObject:
    box: Box3D
    distance_m: float
    footprint_ego: np.ndarray
    track_id: str = ""
    velocity_ego: np.ndarray | None = None
    speed_mps: float = 0.0
    future_xy_ego: np.ndarray | None = None
    position_cov_ego: np.ndarray | None = None
```

Pass it through in `_tracked_world_object`:

```python
def _tracked_world_object(tracked: TrackedObject) -> WorldObject:
    box = tracked.box
    return WorldObject(
        box=box,
        distance_m=float(np.linalg.norm(box.center_ego[:2])),
        footprint_ego=box.corners_ego[:, :2].copy(),
        track_id=tracked.track_id,
        velocity_ego=tracked.velocity_ego,
        speed_mps=tracked.speed_mps,
        future_xy_ego=tracked.future_xy_ego,
        position_cov_ego=tracked.position_cov_ego,
    )
```

Rewrite `_draw_object_motion` to follow the curved forecast and draw the 1σ ellipse:

```python
def _draw_object_motion(
    image: np.ndarray,
    objects: list[WorldObject],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: float,
    scale: int,
    color: tuple[int, int, int] = (40, 220, 255),
    min_speed_mps: float = 0.5,
) -> int:
    """Curved future path + ghost footprint + 1-sigma uncertainty ellipse. Returns moving count."""
    moving = 0
    for obj in objects:
        if obj.velocity_ego is None or obj.speed_mps < min_speed_mps:
            continue
        if obj.future_xy_ego is None or not len(obj.future_xy_ego):
            continue
        moving += 1
        center_xy = obj.box.center_ego[:2]
        path_xy = np.vstack([center_xy, obj.future_xy_ego])
        path = ego_xy_to_bev_pixels(path_xy, x_range, y_range, resolution, scale=scale)
        cv2.polylines(image, [path.reshape((-1, 1, 2))], False, color, 2, cv2.LINE_AA)
        cv2.arrowedLine(image, tuple(path[-2]), tuple(path[-1]), color, 2, tipLength=0.3, line_type=cv2.LINE_AA)

        ghost = obj.footprint_ego + (obj.future_xy_ego[-1] - center_xy)
        gp = ego_xy_to_bev_pixels(ghost, x_range, y_range, resolution, scale=scale)
        cv2.polylines(image, [gp.reshape((-1, 1, 2))], True, color, 1, cv2.LINE_AA)

        _draw_cov_ellipse(image, center_xy, obj.position_cov_ego, x_range, y_range, resolution, scale, color)

        label = f"{obj.speed_mps:.1f}m/s"
        cv2.putText(image, label, (int(path[-1][0]) + 3, int(path[-1][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)
    return moving


def _draw_cov_ellipse(image, center_xy, cov, x_range, y_range, resolution, scale, color):
    if cov is None:
        return
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-6, None)
    angle = np.linspace(0, 2 * np.pi, 24)
    unit = np.stack([np.cos(angle), np.sin(angle)])
    ring = (vecs @ (np.sqrt(vals)[:, None] * unit)).T + center_xy  # 1-sigma
    pix = ego_xy_to_bev_pixels(ring, x_range, y_range, resolution, scale=scale)
    cv2.polylines(image, [pix.reshape((-1, 1, 2))], True, color, 1, cv2.LINE_AA)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_world_model_viz.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add fsd/world_model.py tests/test_world_model_viz.py
git commit -m "feat: curved forecast path + 1-sigma uncertainty ellipse in BEV"
```

---

## Task 9: Full suite + empirical re-validation on a clip

**Files:** none (verification only)

- [ ] **Step 1: Run the whole test suite**

Run: `python -m pytest -q`
Expected: PASS (ekf, tracking, world-model viz, and any pre-existing tests)

- [ ] **Step 2: Run the BEV world model on a short sequence**

Use the existing world-model entry point with both the annotation (GT oracle) and prediction loaders wired (same invocation used to produce the "~0.25 m/s vs oracle" result; check `fsd/world_model.py` consumers / README for the exact command on this machine). Run ~300 frames.

Confirm by observation:
- Moving-object speeds are visually stable frame-to-frame (less jitter than the finite-difference version).
- Forecast paths **curve** for turning vehicles instead of shooting straight.
- Coasted tracks persist briefly through dropped detections, then disappear.

- [ ] **Step 3: Spot-check filtered vs oracle speed**

The GT tracker is the oracle. After a track has been alive ≳1 s, its filtered `speed_mps` should agree with the finite-difference ground-truth speed to within the previously reported tolerance on straight segments, and remain stable through turns.

- [ ] **Step 4: Commit any tuning changes**

If `SIGMA_*` / `max_age_s` need adjustment from observation, change the constants in `fsd/tracking.py` only, re-run `python -m pytest -q`, and commit:

```bash
git add fsd/tracking.py
git commit -m "chore: tune CTRV filter noise/max-age from clip validation"
```

---

## Self-Review Notes

- **Spec coverage:** CTRV model + ω→0 limit (T1), analytic Jacobian (T1–T2), heading measurement with yaw wrapping (T3), curved forecast (T4), Track/measurement helpers (T5), ego-frame read-out incl. covariance (T6), predicted-position gating + coasting + max-age retirement for both trackers (T7), BEV curved path + ellipse (T8), test rewrite + empirical re-validation + tuning (T7, T9). All spec sections map to a task.
- **Naming consistency:** `CtrvEkf.{predict,update,forecast,analytic_F}`, `Track.{predict,correct,global_xy}`, module fns `measurement_global`, `track_to_object`, `_coasted_box`, `_global_to_ego_xy`; constants `SIGMA_POS/SIGMA_YAW/SIGMA_A/SIGMA_YAW_ACCEL`, `HORIZONS_S`, `MIN_SPEED_MPS`, `max_age_s` (ctor kwarg). Used consistently across tasks.
- **No placeholders:** every code step is complete and runnable.
```
