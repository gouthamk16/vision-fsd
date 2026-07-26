# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## Project Overview

Research stack for vision-based autonomous driving. Two independent pipelines
live here — do not confuse them:

| | `fsd/` | `monocular_vision/` |
|---|---|---|
| Input | nuScenes: 6 surround cameras + LiDAR | single dashcam video file |
| Entry point | `python -m fsd.visualize` | `python main.py <video>` |
| Status | active | legacy, maintained but not extended |

New work goes in `fsd/`. `monocular_vision/` is kept because it runs and the
depth/segmentation results are still cited in the README, not because it is
being developed.

## Environment

Python 3.12, CUDA 12.8, an activated `.venv`:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

`fsd/centerpoint_export.py` is the one exception: it needs mmdet3d +
nuscenes-devkit against torch 2.1/cu121, which conflicts with the pins above,
so it runs in a separate `.venv-mmdet3d`. It is not covered by
`requirements.txt` and cannot be imported from the main env.

nuScenes data is read from `D:/nuscenes`, override with `NUSCENES_ROOT`.

Weights: `yolo26n.pt` must be at the repo root. DepthAnything V2 and YOLOPv2
download themselves on first run. The Lift-Splat-Shoot checkpoint is manual —
see the README table — and is passed via `--lss-weights`.

## Architecture — `fsd/` (nuScenes)

`fsd/visualize.py` is the single entry point. Every other module in the
package is a library; none of them have their own CLI, by design. Adding a
capability means adding a view to `visualize.py`, not a new `__main__`.

The dependency order is the thing worth knowing, since it is not obvious from
filenames and most modules sit in the middle of it:

```
data.py              nuScenes loader: scenes, samples, calibration, ego pose
  |
lidar_projection.py  geometry primitives (transform_points,
  |                  quaternion_to_rotation_matrix). Reused by ~8 modules.
  |
bev.py               rasterize a LiDAR sweep to an ego-frame BEV image
bev_tensor.py        2.5D variant: per-cell density + min/max/mean/range height
  |
occupancy.py         temporal log-odds fusion across keyframes (stateful)
  |
world_model.py       unified state: occupancy + height + ego + object
  |                  footprints + per-object velocity (tracking.py)
  |
motion_planning/     consumer: samples trajectories against that world
```

Sitting outside that chain: `lss.py` (camera-only BEV via Lift-Splat-Shoot),
`nuscenes_map.py` (HD map background), `fusion_detect.py` (camera+LiDAR
frustum fusion), `object_detection.py` (GT/prediction box loading + drawing),
`contact_sheet.py` (six-camera tiling).

Views: `cameras`, `lidar`, `bev`, `lss_bev`, `lss_lidar_bev`, `occupancy_bev`,
`height_bev`, `objects_bev`, `objects_cameras`, `planner_bev`,
`planner_camera`, `world_bev`, `all`.

## Architecture — `monocular_vision/` (legacy)

`main.py` → `fsd.driver()` → `FrameProcessor.process()` per frame. The driver
handles video I/O and resamples input above 45 FPS down to a 20 FPS target.

Four subsystems run per frame inside `FrameProcessor`:

| Module | Class | Output |
|---|---|---|
| `detect.py` | `VehicleTracker` | YOLO + ByteTrack 2D boxes, track IDs, class locked per track |
| `depth.py` | `MonocularDepthEstimator` | DepthAnything V2 metric depth, metres |
| `segment.py` | `DrivableAreaSegmentor` | YOLOPv2 binary drivable-area mask |
| `vision.py` | `VisualOdometry` | SIFT → FLANN → essential matrix pose |

BEV rendering in `detect.py` projects 2D boxes plus per-object depth into a
320×440 minimap covering ~83 m forward and ±32 m lateral. Per-class depth
extents are hardcoded priors (Car 1.8 m, Truck 12 m, Bus 2.2 m,
Bicycle/Motorcycle 0.4–4.5 m). Tracked COCO classes: Bicycle(1), Car(2),
Motorcycle(3), Bus(5), Truck(7).

Outputs land in `outputs/`, logs in `logs/`. `LOGGING_LEVEL` controls
verbosity (default INFO).

## Tests

`pytest tests/` — 51 tests, mostly covering `motion_planning/` and
`tracking.py`. There is no linter or CI configured.

Tests are pure-Python and need neither a GPU nor the nuScenes dataset. Changes
touching rendering or the pipelines should additionally be validated by
actually running them:

```bash
python -m fsd.visualize --view world_bev --frames 5 --save
python main.py data/test_nyc.mp4 --stream --frames 60
```

## Code standards

- Simplest correct solution over the extensible one. A function earns its
  existence by being reused or by making the code clearer.
- Files under ~300 lines, functions under ~30. These are the target for new
  and modified code, not a description of the current state: the existing tree
  has 6 files over 300 lines (`build_whitepaper.py` 954, `data.py` 600,
  `lss.py` 545, `object_detection.py` 532, `visualize.py` 456, `detect.py` 305)
  and 65 functions over 30 lines, the worst being `build()` 500,
  `run_visualizer()` 236 and `driver()` 161. Don't grow them; split when you
  have a reason to touch one. A repo-wide refactor to these limits has not
  been done and would need test coverage on the render paths first.
- No comments restating what the code does. Comment the non-obvious only: why
  a threshold is that value, a workaround for a library bug, an invariant not
  visible locally.
- Duplicated logic gets extracted at 3+ occurrences. Two usually don't yet.
- Data crossing a module boundary is a typed structure, not a loose dict. The
  `@dataclass` types in `fsd/data.py`, `object_detection.Box3D` and
  `motion_planning/state.py` are the pattern to follow; nuScenes JSON records
  stay dicts only until they are parsed into one of those.
- Explicit error handling where the failure is actionable, not a blanket
  try/except several layers up. `FrameProcessor.__init__` is the right shape —
  it catches per subsystem so a missing depth model degrades instead of
  killing the pipeline. The broad `except Exception` around the whole
  per-frame body in `process_frame.py` is the wrong shape; don't copy it.
- Imports at module top level. The one accepted exception is a heavy or
  environment-specific ML import deferred into its function, and only with the
  reason stated inline: `centerpoint_export.py` (mmdet3d exists only in
  `.venv-mmdet3d`) and `fusion_detect.py` (`ultralytics` deferred so importing
  the module doesn't load YOLO).
- No throwaway scripts committed. No model weights, video, or large binaries —
  they are gitignored; document how to fetch them instead.
- No stray `TODO` comments. Do it, or write it down outside the code.
- Match the existing style of a file you're editing even if you'd choose
  differently. Don't refactor adjacent code that isn't part of the task —
  every changed line should trace back to the task at hand.
- No temporary fix for a root cause you understand. If it can't be fixed
  properly now, say so explicitly rather than papering over it.
- Turn a task into a verifiable goal before starting, and don't call it done
  without proving it: run `pytest tests/`, and for render or pipeline changes
  actually run the thing and show the output.
- Conventional commits (`feat:`, `fix:`, `refactor:`, `chore:`, `docs:`), one
  logical change each.
- Never claim what code does without reading it.

There is no linter or CI here. `ruff` and `mypy` are the tools these standards
assume; until they are wired up, the checks above are manual.
