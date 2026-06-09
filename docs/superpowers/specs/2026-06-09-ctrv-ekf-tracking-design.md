# CTRV EKF for object tracking — design

Date: 2026-06-09
Status: approved for spec review

## Motivation

Object velocity in the BEV world model is currently a single-step finite
difference of global position (`fsd/tracking.py`), extrapolated with a constant
velocity. Two problems:

1. The per-frame velocity is noisy — a raw difference of two detector positions.
2. Future positions are always straight lines, so turning vehicles are predicted
   to drive off their actual path.

Replace the finite-difference + constant-velocity step with a per-track Extended
Kalman Filter on a Constant Turn-Rate and Velocity (CTRV) model. CTRV makes the
predictions curved (correct for turning vehicles) and the filter denoises the
velocity readout. The README "Todo" calls for exactly this.

## Motion model

Per-track state in the **global frame**:

```
x = (px, py, v, yaw, yaw_rate)
```

CTRV process model (nonlinear → genuine EKF), with ω = yaw_rate:

```
px'       = px + (v/ω)(sin(yaw + ω·dt) − sin yaw)
py'       = py + (v/ω)(−cos(yaw + ω·dt) + cos yaw)
v'        = v
yaw'      = yaw + ω·dt
yaw_rate' = ω
```

The ω → 0 (straight-line) limit is handled with an explicit branch:

```
px' = px + v·cos(yaw)·dt
py' = py + v·sin(yaw)·dt
```

Process noise comes from random linear acceleration `σ_a` and yaw acceleration
`σ_ω̇` (the standard CTRV Q construction). The predict-step Jacobian `F` is
computed analytically for both the general and the ω → 0 branch.

## Measurement

Each detection contributes position and heading:

```
z = (px, py, yaw)   # global frame
```

- `px, py` from `box.center_ego` transformed to global (same path as the current
  `_global_xy`).
- `yaw` = `box.yaw_ego + ego_yaw`, where `ego_yaw` is extracted from the ego pose
  rotation.

The measurement model is **linear** (`H` selects px, py, yaw from the state), so
the only nonlinearity is in the predict step. Angle residuals (`yaw`) are wrapped
to `(−π, π]` before applying the gain.

## Architecture

### `fsd/ekf.py` (new)

`CtrvEkf` — pure filter math, no nuScenes dependencies:

- holds `x` (5,) and `P` (5×5)
- `predict(dt)` — CTRV propagation + analytic `F`, adds `Q(dt)`
- `update(z)` — linear `H`, angle-wrapped innovation, Joseph or standard form
- `forecast(horizons_s)` — forward-integrates the CTRV model from the current
  state, returning (K, 2) future centers in the global frame (curved path)

Unit-testable in isolation against synthetic trajectories.

### `fsd/tracking.py` (refactored)

A `Track` wraps one `CtrvEkf` plus `track_id`, `class_name`, and a `misses`
counter. The two existing trackers keep their association strategies but become
thin lifecycle managers over a dict of `Track`:

- `GtVelocityTracker` — association by stable nuScenes `instance_token` (oracle).
- `PredictionVelocityTracker` — greedy nearest-by-class association for
  identity-free detector boxes.

Per `update(frame, boxes)` both do:

1. `predict(dt)` every existing track to the current timestamp.
2. Associate detections to tracks. Gating uses the EKF's **predicted** position
   (tighter and more correct than gating on the last raw position).
3. `update(z)` matched tracks; reset their `misses` to 0.
4. Unmatched tracks: predict-only **coast**, increment `misses`; retire when the
   coasted age exceeds `max_age` (~0.5 s).
5. Spawn a new `Track` for each unmatched detection (init `v=0`, `yaw_rate=0`,
   large `P` on `v` and `yaw_rate`, `yaw` from the measurement).

`_make_tracked` is removed; `TrackedObject` is read directly from each `Track`.

## Outputs (`TrackedObject`)

- `velocity_ego` (2,) / `speed_mps` — filtered, derived from `v` and `yaw`
  rotated into the current ego frame. Denoised vs the current raw difference.
- `future_xy_ego` (K, 2) — **curved** CTRV forecast at 1/2/3 s, transformed from
  global into the current ego frame.
- `position_cov_ego` (2×2) *(new)* — the position block of `P` rotated into the
  ego frame, for a 1σ uncertainty ellipse in the BEV. The natural EKF payoff for
  "path tracking".

## Visualization (`fsd/world_model.py`)

`_draw_object_motion`:

- draw the future ghost footprint along the **curved** `future_xy_ego` (use the
  last forecast point for the ghost offset; the arrow follows the forecast
  direction rather than a straight `velocity_ego` extension).
- add a faint 1σ uncertainty ellipse from `position_cov_ego` per moving track.

Display gating (`min_speed_mps = 0.5`) is unchanged.

## Validation

The current `tests/test_tracking.py` asserts *exact* one-step finite-difference
velocity (e.g. `speed_mps == 5.0` after one update). That contract no longer
holds — a filter converges over frames rather than matching in one step. Tests
are rewritten:

- **`tests/test_ekf.py`** (new) — drive `CtrvEkf` with synthetic straight-line
  and constant-turn trajectories plus Gaussian measurement noise; assert the
  state converges within tolerance after K frames, and that the curved forecast
  tracks a turning trajectory markedly better than a straight constant-velocity
  extrapolation.
- **`tests/test_tracking.py`** (rewritten) — cover association, predicted-position
  gating, coasting / `max_age` retirement, and ego-frame rotation; assert
  filtered estimates **converge** (tolerance-based), not exact-match.
- Re-run the empirical oracle comparison. CTRV should match the old constant-
  velocity result on straight motion and beat it on turns.

## Tuning defaults

```
σ_pos     ≈ 0.5 m       # detector localization noise
σ_yaw     ≈ 0.1 rad
σ_a       ≈ 2.0 m/s²    # process: linear accel
σ_ω̇      ≈ 0.5 rad/s²  # process: yaw accel
max_age   = 0.5 s
min_speed = 0.5 m/s     # display gating, unchanged
```

Init on first sighting: `x = (px, py, 0, yaw_meas, 0)`, large `P` on `v` and
`yaw_rate`.

## Out of scope

- Multi-hypothesis / probabilistic data association (greedy stays).
- Interacting Multiple Model (IMM) filters — single CTRV model only.
- Integrating the EKF into the legacy monocular pipeline (`monocular_vision/`);
  this targets the nuScenes BEV world model.
