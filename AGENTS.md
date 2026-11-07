# AGENTS.md

Guidance for Codex (Codex.ai/code) working in this repository.

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
`requirements.txt` and cannot be imported from the main env. It is one of two
modules with their own CLI, because it cannot run in the main env; the other
is `fsd/nmp/train.py`, because training is not a view.

nuScenes data is read from `C:/nuscenes`, override with `NUSCENES_ROOT`. The
flat devkit layout is expected (`samples/`, `sweeps/`, `maps/`,
`v1.0-trainval/`). Only scenes whose blobs are extracted are listed, so scene
indices count available scenes. The first run indexes the large metadata
tables once into `~/.cache/vision-fsd/`; extracting another blob part
invalidates that cache automatically.

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
nuscenes_index.py    one streaming pass over the big metadata tables, cached
  |
data.py              Pose (typed rigid transform), Camera/Lidar/SurroundFrame,
  |                  scene + sweep iteration. Frames carry the LIDAR_TOP ego
  |                  pose as their reference: annotations are defined there.
  |
ego_motion.py        ego footprint (rear-axle origin) and EgoState
lidar_projection.py  multi-sweep ego-frame points (self-returns removed),
  |                  camera projection
  |
bev.py               BevGrid: the single BEV pixel convention, resampling,
  |                  and anti-aliased vector drawing at view resolution
  |
bev_tensor.py        per-cell height statistics on a BevGrid
occupancy.py         temporal log-odds fusion with ray-cast free space
  |
world_model.py       occupancy + ego state + tracked objects (tracking.py: one
                     CTRV Kalman filter per track, ekf.py)
```

Rasters are computed on a coarse grid (`--bev-resolution`, 0.25 m) and views
render the same extent on a fine grid (`--bev-ppm`, 10 px/m), drawing
points, boxes, map and text directly at that resolution. Never draw on a
raster and upscale it; that is what made the old views blurry.
`planner_bev` is the one exception: a matplotlib figure (ego-centred crop
plus the cost at 1/2/3 s) built for reading the planner, not for matching
the other views' extent. It still never upscales.

`visualize.py` computes per-frame products (points, occupancy, detections,
LSS output, map background) once and shares them across views; a view is an
entry in its `RENDERERS` table.

Sitting outside that chain: `draw.py` (headers, labels, blending),
`contact_sheet.py` (six-camera tiling with per-tile annotation),
`nuscenes_map.py` (HD map layers), `lss.py` (camera-only BEV via
Lift-Splat-Shoot), `fusion_detect.py` (camera+LiDAR frustum fusion),
`object_detection.py` (GT/prediction boxes and drawing).

Views: `cameras`, `lidar`, `bev`, `lss_bev`, `lss_lidar_bev`, `occupancy_bev`,
`height_bev`, `objects_bev`, `objects_cameras`, `world_bev`, `planner_bev`,
`all` (LSS views need `--lss-weights`, `planner_bev` needs `--nmp-weights`;
`all` skips whichever is missing).

## Architecture — `fsd/nmp/` (neural motion planner)

NMP (Zeng et al., CVPR 2019): one conv net over 10 LiDAR sweeps + 8 HD-map
channels outputs vehicle detections with 3 s forecasts and a cost volume (one
cost map per future 0.5 s step); planning picks the cheapest of 1000 sampled
trajectories. Paper summary: `knowledge/summary_neural_motion_planner.md`;
plan and rulings: `docs/superpowers/plans/2026-11-04-nmp-planner.md`.

It consumes `data`, `lidar_projection`, `nuscenes_map` and `object_detection`;
nothing imports it except `visualize` (the `planner_bev` view).

```
config.py        NmpConfig: grid (140.8 x 80 m @ 0.2 m), sweeps, horizon, losses
trajectories.py  straight / circle / clothoid sampler (curvature linear in arc length)
samples.py       keyframe -> NmpSample (sparse voxels, map masks, ego future, boxes)
dataset.py       densify cached samples, rasterise future obstacles, left-right mirror augmentation
model.py         PIXOR backbone + centre-heatmap detection/forecast head + cost-volume head
losses.py        CenterPoint heatmap + smooth-L1 perception loss (heading as axis + side), max-margin planning loss
metrics.py       L2 / box collision / off-road, constant-velocity baseline
planner.py       inference: NmpPlanner.plan(frame, lidar, map) -> PlanResult
evaluate.py      validation metrics and the eval report
train.py         CLI: cache / fit / eval
render.py        planner_bev figure: scene + plan vs logged human, cost panels
```

```bash
python -m fsd.nmp.train cache --split val      # ~2 min, ~1k samples
python -m fsd.nmp.train cache --split train    # ~9 min, ~4.4k samples
python -m fsd.nmp.train fit                    # 20 epochs, ~8.5 min each on an 8 GB GPU
python -m fsd.nmp.train fit --resume           # continue an interrupted run from checkpoints/nmp/last.pt
python -m fsd.nmp.train eval --weights checkpoints/nmp/best.pt
python -m fsd.visualize --view planner_bev --nmp-weights checkpoints/nmp/best.pt --scene-name scene-0003 --save
```

The val split is the official nuScenes `val` list (`fsd/nmp/val_scenes.txt`);
train is every other available scene. The cache lives in
`~/.cache/vision-fsd/nmp/`, checkpoints in `checkpoints/nmp/` (gitignored).

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
extents are hardcoded priors (Car 4.5 m, Truck 8.5 m, Bus 12 m,
Bicycle 1.8 m, Motorcycle 2.2 m). Tracked COCO classes: Bicycle(1), Car(2),
Motorcycle(3), Bus(5), Truck(7).

Outputs land in `outputs/`, logs in `logs/`. `LOGGING_LEVEL` controls
verbosity (default INFO).

## Tests

`pytest tests/` covers the data layer, BEV geometry, occupancy, tracking,
detection geometry, the map renderer, the visualizer's view selection, and
the NMP sampler, samples, dataset, model, losses, metrics and a CPU overfit
run of training.
There is no linter or CI configured.

Tests are pure-Python and need neither a GPU nor the nuScenes dataset. Changes
touching rendering or the pipelines should additionally be validated by
actually running them:

```bash
python -m fsd.visualize --view all --frames 5 --save
python main.py data/test_nyc.mp4 --stream --frames 60
```

## Code standards

- Simplest correct solution over the extensible one. A function earns its
  existence by being reused or by making the code clearer.
- Files under ~300 lines, functions under ~30. Don't grow the exceptions
  below; split when you
  have a reason to touch one. `fsd/` is within them apart from `lss.py`'s
  faithful port of the upstream model; the remaining offenders are
  `docs/build_whitepaper.py` (`build()` 500 lines) and the legacy
  `monocular_vision/` (`driver()` 167 lines).
- No comments restating what the code does. Comment the non-obvious only: why
  a threshold is that value, a workaround for a library bug, an invariant not
  visible locally.
- Duplicated logic gets extracted at 3+ occurrences. Two usually don't yet.
- Data crossing a module boundary is a typed structure, not a loose dict. The
  `@dataclass` types in `fsd/data.py` (`Pose`, the frame types) and
  `object_detection.Box3D` are the pattern to follow; nuScenes JSON records
  stay dicts only until they are parsed into one of those.
- Explicit error handling where the failure is actionable, not a blanket
  try/except several layers up. `FrameProcessor.__init__` is the right shape —
  it catches per subsystem so a missing depth model degrades instead of
  killing the pipeline. A broad `except Exception` around a whole per-frame
  body is the wrong shape.
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
