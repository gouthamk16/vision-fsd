# Autonomous Driving System — Architectural Roadmap

## The Biggest Architectural Weakness Right Now

our system is still **frame-centric**.

Meaning:
- process frame,
- render outputs,
- move to next frame.

Real autonomy stacks are **stateful**.

The vehicle maintains:
- persistent world understanding,
- object memory,
- motion estimates,
- future predictions,
- map priors.

This is the major shift we should make next.

---

## What I Would Do Next (In Order)

### Priority 1 — Temporal Occupancy Mapping

This is the **single best next step** for our project.

we already have:
- depth,
- segmentation,
- odometry,
- BEV.

Now combine them **across time**.

Right now our BEV is probably: *"current frame projection."*

Instead: **Build a persistent world map.**

As the car moves:
- accumulate occupancy information,
- stabilize BEV using odometry,
- maintain rolling world representation.

Think: *"What does the world around me look like over the last 5 seconds?"*

For the near-term autonomy stack, the world model should be **2D/2.5D BEV**, not a full 3D voxel map:
- 2D occupancy answers the planning question: *can the ego vehicle occupy this x-y space safely?*
- LiDAR already gives height (`z`), so each BEV cell can also store height summaries: max height, min height, mean height, height range, and point density.
- 3D object detections should be preserved as 3D boxes, but planning consumes their top-down footprints: x, y, width, length, yaw, velocity.

Full 3D voxel occupancy is deferred until we actually need vertical volumetric reasoning such as overpasses, multi-level roads, overhead obstacles, or semantic 3D occupancy benchmarks.

#### Why This Matters

This unlocks:
- obstacle motion understanding,
- trajectory planning,
- path prediction,
- free-space estimation,
- future occupancy prediction.

> Without temporal fusion, we don't really have autonomy. we just have perception overlays.

#### What This Would Look Like

we'd maintain:
```
global_occupancy_grid
tracked_objects
ego_pose_history
```

Each frame:
1. estimate ego motion
2. warp previous BEV into current coordinate frame
3. fuse new occupancy estimates
4. decay uncertain regions
5. update tracked object states

Now we have: **a continuously evolving world model.** That's real autonomy infrastructure.

---

### Priority 2 — Lane Topology + Road Graphs

our current segmentation is good. But segmentation alone is not enough.

we need:
- lane centerlines,
- lane connectivity,
- road boundaries,
- driving corridors.

This matters because planning happens relative to: lanes, reachable trajectories, and traffic flow.

#### What To Build

Not just *"lane pixels."* Instead:
- extract lane splines,
- fit curves,
- estimate lane direction,
- construct lane graph.

This is much more advanced and much more useful.

---

### Priority 3 — Object Motion Prediction

Right now: we detect objects.

Next: **predict where they will be.**

For each tracked object, estimate:
- velocity,
- heading,
- future trajectory.

Even simple Kalman filtering is huge here.

#### Start With

For every object: `[x, y, vx, vy]`

Then predict:
```
future_position = current + velocity * dt
```

This alone transforms our project dramatically.

---

### Priority 4 — Planning

This is the biggest conceptual leap.

we now need: *"Given the world state, where should the vehicle go?"*

This means:
- collision avoidance,
- trajectory generation,
- path smoothing,
- speed planning.

#### Don't Start With Neural Planning

Start classical. Use:
- A*
- spline planners
- pure pursuit
- MPC (later)

**Why?** Because it's debuggable, interpretable, and educational.

---

## our BEST Next Major Milestone

> **"Navigate a CARLA town autonomously using monocular perception only"**

Pipeline:

```
Camera
  ↓
Perception
  ↓
Temporal Occupancy Map
  ↓
Lane Graph
  ↓
Object Prediction
  ↓
Trajectory Planner
  ↓
Controller
  ↓
Vehicle Actuation
```

That becomes a **legitimate autonomous driving system.**

---

## What I Would NOT Focus On Yet

Don't:
- train giant end-to-end transformers,
- imitate Tesla V12 immediately,
- build giant multimodal LLM planners,
- over-focus on huge architectures.

we still need **system-level fundamentals.**

Recruiters and researchers are often more impressed by:
- a robust planner,
- good world modeling,
- proper controls,

...than a giant black-box model.

---

## My Actual Recommendation — Full Roadmap

### Phase 1 — World Modeling
- [ ] Temporal BEV fusion
- [ ] Occupancy grids
- [ ] 2.5D LiDAR height channels
- [ ] Lane topology
- [ ] Object velocity estimation

### Phase 2 — Decision Layer
- [ ] Local trajectory planner
- [ ] Obstacle avoidance
- [ ] Rule-based driving policy

### Phase 3 — Control
- [ ] Steering controller
- [ ] Throttle/brake controller
- [ ] CARLA integration

### Phase 4 — Learning-Based Improvements
- [ ] Pretrained LiDAR 3D detector integration
- [ ] Imitation learning planner
- [ ] Trajectory transformer
- [ ] Occupancy prediction networks

---

# Build Log — nuScenes 360 World Modeling

Phase 1 is built on [nuScenes](https://www.nuscenes.org/) (six surround cameras + LiDAR + 3D labels + HD maps) rather than stretching the monocular pipeline. LiDAR gives metric depth and the dataset ships exact 3D boxes and ego poses, so monocular depth, 2D back-projection, and SIFT odometry are deliberately not used on this arm.

## Dataset rule

nuScenes is an external, **read-only** dataset. Root defaults to `D:/nuscenes`; override with `NUSCENES_ROOT`. Never copy samples, sweeps, maps, or metadata into the repo or onto `C:/`. Metadata is read straight from the JSON tables — **no `nuscenes-devkit` runtime dependency** anywhere in `fsd/`.

The drive is sometimes unmounted in the shell; renders that hit `D:/nuscenes` only run when it is mounted.

## What's built

`fsd.visualize` is the single entry point. Pick a view with `--view`, scenes with `--scenes N` or `N-M`, length with `--frames N|all`, and `--save` or `--stream`.

| Capability | Module | View(s) |
|---|---|---|
| Read-only scene loader (6 cams + LiDAR, lazy image loads) | `fsd/data.py` | — |
| Six-camera contact sheet | `fsd/contact_sheet.py` | `cameras` |
| LiDAR projected into the camera images, coloured by depth | `fsd/lidar_projection.py` | `lidar` |
| Ego-frame LiDAR BEV rasterizer | `fsd/bev.py` | `bev` |
| nuScenes GT 3D boxes in BEV and reprojected onto cameras | `fsd/object_detection.py` | `gt_bev`, `box_cameras` |
| Detector prediction vs GT in BEV (JSON adapter) | `fsd/object_detection.py` | `pred_bev`, `compare_bev` |
| Camera-only LSS BEV vehicle seg + HD-map background | `fsd/lss.py`, `fsd/nuscenes_map.py` | `lss_bev`, `lss_lidar_bev` |
| Temporal log-odds occupancy fusion | `fsd/occupancy.py` | `occupancy_bev` |
| 2.5D BEV tensor: per-cell density + min/max/mean height + range | `fsd/bev_tensor.py` | `height_bev` |
| Camera object detection, LiDAR-ranged (frustum fusion) | `fsd/fusion_detect.py` | `objects_bev`, `objects_cameras` |
| CenterPoint LiDAR 3D detection (mmdet3d, isolated env) | `fsd/centerpoint_export.py` (+ `.venv-mmdet3d`) | `pred_bev`, `compare_bev` |

Notes:
- `--sequence sweeps` gives ~12 Hz smooth camera video (intermediate frames between keyframes); everything else uses 2 Hz keyframes.
- `--view all` renders every view at once.

## Milestone notes

### Camera-only BEV — Lift-Splat-Shoot (`fsd/lss.py`)

NVIDIA's Lift-Splat-Shoot (ECCV 2020) BEV vehicle-segmentation model, the first learned BEV head in the project.

- 1:1 port of `LiftSplatShoot`, `CamEncode`, `BevEncode`, `Up`, `QuickCumsum`, and the grid-config helpers from the upstream repo (cloned to `external/lift-splat-shoot/`, git-ignored).
- `LSSInference` consumes our `SurroundFrame` directly: builds the six per-camera intrinsics/extrinsics and the eval-time resize+center-crop (to 128x352) used in upstream `data.py`.
- Output is 200x200 @ 0.5 m/cell, rendered in the same orientation as our LiDAR BEV (ego centred, forward = top). `lss_lidar_bev` blends the mask onto the LiDAR BEV — this requires resampling the LSS grid to the LiDAR canvas dims (an early bug pasted it at 2x the wrong scale).
- Added dependency: `efficientnet_pytorch==0.7.0`. ImageNet weights are **not** downloaded (`from_name`); the LSS checkpoint overwrites the trunk.
- Weights: [Google Drive](https://drive.google.com/file/d/1bsUYveW_eOqa4lglryyGQNeC4fyQWvQQ/view?usp=sharing) → `models/`, pass via `--lss-weights`.

```powershell
.\.venv\Scripts\python.exe -m fsd.visualize --dataroot D:/nuscenes --scenes 0 --frames 40 --view lss_bev --save --bev-scale 2 --lss-weights models/model525000.pt --output outputs/nuscenes_scene0_lss_bev.mp4
```

**HD-map background** (`fsd/nuscenes_map.py`): the LSS paper's road backdrop comes from the nuScenes map_expansion vector maps. The archive is extracted to `D:/nuscenes/maps/expansion/` (boston-seaport + three Singapore JSONs). nuScenes itself is mixed — 467 Boston scenes, 383 Singapore — and per-scene location is read from `log.json`. We parse the expansion JSON directly (no devkit) and draw road/lane fill, dividers, and ped-crossings in the LSS palette. Auto-enabled for LSS views when the folder exists; disable with `--no-map`.

**"Shoot" is not in the repo:** the released LSS code ships only the lift+splat vehicle-seg head. The cost-map head and template trajectory shooting (paper §4.4) are not included, so they're not in our port.

### Temporal occupancy fusion (`fsd/occupancy.py`) — Priority 1, done

A rolling ego-frame **log-odds** occupancy grid, fused across keyframes — the move from per-frame perception to a stateful world model.

Per keyframe:
1. **Warp** the previous grid into the current ego frame via the relative SE(2) from nuScenes ego poses. The pixel affine `M = M2P · SE2_{cur→prev} · P2M` is fed to `cv2.warpAffine` with `WARP_INVERSE_MAP` (OpenCV otherwise inverts `M` itself).
2. **Decay** the log-odds toward 0 (unknown).
3. **Update** from LiDAR evidence using a **height split**: returns below `ground_height` (0.3 m, ego frame) mark cells free, taller returns mark them occupied. (An earlier polar nearest-hit free-space polygon was dropped — ground returns are the nearest hit in every direction, so it both collapsed the free region and painted the road as occupied.)

Verified on synthetic cases: 2 m forward motion moves a 20 m wall to 18 m; +90° yaw moves a point straight ahead to the right. Keyframes only; the mapper resets at each scene boundary.

```powershell
.\.venv\Scripts\python.exe -m fsd.visualize --dataroot D:/nuscenes --scenes 0 --frames all --view occupancy_bev --save --fps 4 --bev-scale 2 --output outputs/nuscenes_scene0_occupancy.mp4
```

The navy free corridor trailing the ego is the visible proof that prior-frame evidence is being warped and accumulated.

### 2.5D BEV height-channel tensor (`fsd/bev_tensor.py`) — Phase 1, done

Hardens the BEV from "occupied/free" into a multi-channel world-model tensor. Every channel is computed from the LiDAR sweep alone (no labels, no map) so it is deployable on a real vehicle. Per cell: `density` (point count), `max_height`, `min_height`, `mean_height`, `height_range` (= max − min). `height_range` is the useful one — flat road is ~0, cars/walls are large — so it separates drivable surface from obstacles without any learning.

`BevTensor.stack()` returns an `(H, W, 5)` array; `compute_bev_height_channels` does the rasterization (vectorized `np.add.at` / `np.maximum.at`). The `height_bev` view tiles all five channels for inspection. Verified synthetically: a 2.5 m wall column reads ~2.5 m height range, flat ground reads ~0.

```powershell
.\.venv\Scripts\python.exe -m fsd.visualize --dataroot D:/nuscenes --scenes 0 --frames 40 --view height_bev --save --bev-resolution 0.25 --output outputs/nuscenes_scene0_height_bev.mp4
```

### Frustum-fusion object detection (`fsd/fusion_detect.py`) — dynamic layer, done

The dynamic-object layer, built **hybrid** by deliberate choice: cameras detect, LiDAR ranges. Decision context is in the chat log - we keep LiDAR for geometry (occupancy/height) while this branch does object detection from vision. CenterPoint is the canonical learned LiDAR 3D detector path.

Pipeline per keyframe:
1. YOLO (`yolo26n.pt`, already in repo) runs on each of the six camera images → 2D boxes + class. This is the detection decision — camera-only.
2. The 2D box defines a viewing **frustum**. Project the LiDAR sweep into that camera; the points landing inside the box are the object's returns. Drop ground (`z > 0.3`), keep the nearest range cluster (rejects background seen through/around the box), and that gives the object's metric position — no depth guessing.
3. Build an ego-frame `Box3D`: position from the cluster, footprint from a class size prior, heading from the cluster's dominant horizontal axis (PCA), rested on the ground plane (`z = height/2`). These are the "2D corner coords for BEV" — `Box3D.corners_ego`.
4. Dedup objects seen by two overlapping cameras (greedy, by class + center distance).

Views: `objects_bev` (ranged boxes on the LiDAR BEV) and `objects_cameras` (boxes reprojected onto the six images). CLI: `--yolo-weights`, `--detector-device`.

```powershell
.\.venv\Scripts\python.exe -m fsd.visualize --dataroot D:/nuscenes --scenes 0 --frames 40 --view objects_bev --save --bev-scale 2 --output outputs/nuscenes_scene0_objects_bev.mp4
```

Honest limits: positions are as good as the LiDAR cluster (solid), but class is YOLO/COCO (car/truck/bus/bike/motorcycle/pedestrian only), size is a prior not measured, and heading from a partial-view cluster is rough. Good enough to feed tracking → velocity next. Validated synthetically (ground/background rejection, ground-resting, dedup); full-scene render pending the data drive being mounted.

### CenterPoint LiDAR 3D detector via mmdet3d — working (isolated env)

A real, pretrained 3D detector (not the 2D-YOLO frustum hack). The OpenMMLab stack would not build on the main `.venv` (Py3.12 + Torch 2.7 + CUDA 12.8 — no prebuilt `mmcv`), so it lives in a **separate, pinned env** and feeds boxes to the visualizer through the existing prediction-JSON adapter (`pred_bev` / `compare_bev`). **Working end-to-end on real nuScenes**: scene 0 / 40 frames → 1454 boxes (~36/frame); in `compare_bev` the red predictions sit on top of the green GT boxes with correct headings. Outputs: `outputs/nuscenes_scene0_centerpoint_bev.mp4`, `outputs/nuscenes_scene0_centerpoint_vs_gt.mp4`.

Working install recipe (reproducible — the key was avoiding any from-source `mmcv` build):

```powershell
# 1. isolated Python 3.11 env (main .venv is Py3.12 — incompatible with prebuilt mmcv)
#    got 3.11 via `uv` (uv python install 3.11), then:
<py3.11>\python.exe -m venv .venv-mmdet3d
# 2. Torch 2.1 + cu121 — the combo with prebuilt mmcv wheels (cu121 binaries run on the 12.8 driver)
.venv-mmdet3d\Scripts\python.exe -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
# 3. OpenMMLab via mim (pulls the prebuilt mmcv 2.1.0 wheel — no compile)
.venv-mmdet3d\Scripts\python.exe -m pip install -U openmim numpy==1.26.4
.venv-mmdet3d\Scripts\mim.exe install mmengine "mmcv==2.1.0" "mmdet==3.2.0"
# 4. mmdet3d WITHOUT deps (its lyft-dataset-sdk dep drags an ancient source-built matplotlib)
.venv-mmdet3d\Scripts\python.exe -m pip install --no-deps "mmdet3d==1.4.0"
.venv-mmdet3d\Scripts\python.exe -m pip install numba==0.59.1 llvmlite==0.42.0 plyfile trimesh scikit-image nuscenes-devkit "pandas<2.2"
```

Pinned versions: Python 3.11.15, torch 2.1.0+cu121, mmcv 2.1.0, mmdet 3.2.0, mmdet3d 1.4.0, numpy 1.26.4.

**Voxel upgrade (done):** the higher-accuracy voxel CenterPoint (~0.50 → ~0.65 NDS) needs `spconv` (sparse 3D conv). A prebuilt wheel installs cleanly — no source build:

```powershell
.venv-mmdet3d\Scripts\python.exe -m pip install spconv-cu120   # cu120 kernels run on the 12.8 driver
```

Verified: GPU `SubMConv3d` runs; the voxel0075 + DCN model (DCN ops come from mmcv) does a full forward pass. Use `centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d`. Scene 0 / 40 frames -> 1339 boxes; in `compare_bev` the red predictions sit tightly on the green GT with good heading agreement. Outputs: `outputs/nuscenes_scene0_centerpoint_voxel_vs_gt.mp4`, `outputs/nuscenes_scene0_centerpoint_voxel_cameras.mp4`.

Run (in the mmdet3d env) → export ego-frame boxes to JSON, then view in the main env:

```powershell
.venv-mmdet3d\Scripts\python.exe -m fsd.centerpoint_export --dataroot D:/nuscenes --config checkpoints\centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py --checkpoint checkpoints\centerpoint_0075voxel_*.pth --scene-index 0 --frames 40 --output outputs/centerpoint_scene0.json
.venv\Scripts\python.exe -m fsd.visualize --dataroot D:/nuscenes --scenes 0 --frames 40 --view pred_bev --predictions outputs/centerpoint_scene0.json --save --output outputs/nuscenes_scene0_centerpoint_bev.mp4
```

`fsd/centerpoint_export.py` converts CenterPoint's LiDAR-sensor-frame boxes to ego frame using the LiDAR `calibrated_sensor` extrinsic and writes `{"samples": {sample_token: [...]}}` for `PredictionLoader`.

**Multi-sweep fix (important).** nuScenes CenterPoint is trained on **10 accumulated LiDAR sweeps** with the per-point 5th channel set to a **time delta**. The naive `inference_detector(single .pcd.bin)` only sees one sweep — the config's `LoadPointsFromMultiSweeps` just pads by duplicating it — so the model ran on ~25k points instead of ~270k, causing visibly worse recall/localization and spurious boxes (the "prediction errors" first observed). Fix: `centerpoint_export.py` aggregates the real 10 sweeps itself (nuscenes-devkit transforms, into the current LiDAR frame, with the time channel) and strips `LoadPointsFromMultiSweeps` from the pipeline so it isn't re-padded. Effect on scene 0: ~25k → ~270k points/frame, 1339 → 985 boxes (fewer false positives), and predictions sit tightly on GT in `compare_bev`. Controlled by `--sweeps` (default 10).

## Next

- [x] **Unified BEV world model shell**: `fsd/world_model.py` now combines temporal occupancy, 2.5D LiDAR height channels, ego state, collision grid, GT boxes, and optional prediction boxes into one `BevWorldModel`. Visualize with `--view world_bev`.

- [x] **Object velocity + prediction (GT oracle)** (Priority 3): `fsd/tracking.py` `GtVelocityTracker` finite-differences each object's global position over `instance_token` across keyframes, rotates the velocity into the current ego frame, and constant-velocity-extrapolates to 1/2/3 s. Wired into `BevWorldModel` (`WorldObject.velocity_ego/speed_mps/future_xy_ego`); `world_bev` draws cyan velocity arrows + faded future ghost footprints + speed labels. Validated on scene 0 (frame 0 = 0 m/s, then realistic 5-8 m/s). This is the oracle a learned tracker is checked against.
- [ ] **Object velocity (predicted boxes)** (Priority 3): swap the `instance_token` oracle for greedy nearest-by-class association over CenterPoint predictions to get per-object `[x, y, vx, vy]`; validate the recovered velocities against the GT oracle above. Optional small Kalman smoother if jittery.
- [ ] **Quantify LSS**: vehicle-seg IoU vs the GT BEV mask we already render (paper reports ~33).
- [ ] **Lane topology** (Priority 2): lane centerlines + connectivity into a lane graph from the parsed HD map.
- [ ] **Run + eval the detector**: render `objects_bev`/`objects_cameras` once the drive is mounted; spot-check fusion boxes against GT boxes (eval-only) to gauge position error.
- [ ] **Lane topology** (Priority 2): vision-first (YOLOPv2 lane lines → ego BEV via ground plane), not HD-map lanes.
- [ ] **Planner** (Phase 2): classical/optimization trajectory planner over occupancy + height tensor + lanes + predicted objects. Deferred until the world model is complete enough to plan over.

Object detection is done (camera frustum fusion). The dynamic branch now has GT-oracle velocity + short-horizon prediction; the next step is recovering the same from CenterPoint predictions (own-detection tracking). The static side (occupancy, height, soon lanes) is in good shape.
