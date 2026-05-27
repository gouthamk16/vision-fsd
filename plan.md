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

## Phase 1 Decision — Move From Monocular Frames to 360 World Modeling

### Local Dataset Rule

nuScenes is stored on the data drive and must be treated as an external, read-only dataset root.

Current machine check found the dataset at:

```text
D:/nuscenes
```

The originally mentioned path, `D:/Submariner/nuscenes`, was not present in the shell. We should keep the root configurable as `NUSCENES_ROOT` and never copy samples, sweeps, maps, or metadata into this repo or anywhere on `C:/`.

Expected nuScenes structure:

```text
D:/nuscenes/
  samples/
    CAM_FRONT/
    CAM_FRONT_LEFT/
    CAM_FRONT_RIGHT/
    CAM_BACK/
    CAM_BACK_LEFT/
    CAM_BACK_RIGHT/
    LIDAR_TOP/
    RADAR_*/
  sweeps/
  maps/
  v1.0-trainval/
```

### Recommendation

Phase 1 should not start by training a giant BEV transformer. Start with a small, inspectable 360 perception/world-modeling baseline on nuScenes, then use pretrained BEV/occupancy models as benchmarks.

The first milestone should be:

> Load one nuScenes scene, synchronize the six cameras, project perception into an ego-frame BEV grid, fuse over time using ego poses, and render a rolling 360 occupancy/free-space map.

This gives us the core autonomy infrastructure in a way we can debug visually.

### Phase 1A — Dataset + 360 Frame Abstraction

Build a new `surround_vision` pipeline rather than stretching `monocular_vision` too far.

Core objects:

```text
NuScenesSceneLoader
  -> SurroundFrame
      timestamp
      sample_token
      ego_pose
      cameras[6]
        image_path
        intrinsics
        extrinsics
```

Use `nuscenes-devkit` for metadata and calibration. Images should be loaded lazily from `D:/nuscenes/samples/...` only when needed.

Important outputs:
- six-camera contact sheet for a sample
- calibration sanity check by projecting LiDAR points into camera views
- ego-pose sequence for temporal fusion

### Phase 1B — Geometry Baseline for 360 BEV

**Pivot from the monocular pipeline.** nuScenes ships LiDAR + 3D annotations, which makes monocular depth and 2D YOLO back-projection strictly worse than the sensor truth:

- LiDAR gives metric depth at ~cm accuracy out to ~70m; Depth Anything has ~10–20% relative error that compounds in BEV.
- nuScenes 3D box annotations give exact object pose, size, and yaw — no need to derive 3D from 2D YOLO + noisy depth.
- The standard upgrade path for predictions (later) is a LiDAR 3D detector (CenterPoint / PointPillars), not 2D → depth back-projection.

What we **do** keep from the monocular project:

- Drivable-area segmentation (YOLOPv2), but front-camera only and aimed at lane topology rather than free-space — LiDAR + HD map answer free-space better.
- nuScenes ego pose (replaces our SIFT odometry entirely).

What the geometry baseline does each frame:

1. Load `LIDAR_TOP` points from `D:/nuscenes`.
2. Transform to ego frame using calibrated sensor extrinsics.
3. Rasterize into a shared ego-frame BEV grid (already implemented in `fsd/bev.py`).
4. Overlay nuScenes 3D annotation boxes as the object layer.
5. Optionally run drivable-area on `CAM_FRONT` for a semantic layer.

Monocular depth and 2D detection are deferred to a future camera-only research arm (BEVFormer / LSS direction), not Phase 1.

First BEV grid:

```text
range: x=[-50m, 50m], y=[-50m, 50m]
resolution: 0.25m or 0.5m per cell
channels:
  occupied_prob
  free_prob
  point_density
  max_height
  min_height
  mean_height
  height_range
  drivable_prob
  dynamic_object_prob
  semantic_class_id
  last_observed_time
```

This is a 2D/2.5D world model. The BEV grid is 2D in x-y, but LiDAR `z` values are summarized into height channels. That gives most of the useful vertical awareness without the memory and debugging cost of full 3D voxels.

### Phase 1C — Temporal Occupancy Fusion

Use the official nuScenes ego pose for this stage. Do not rely on monocular visual odometry yet.

Loop:

```text
previous_grid --warp by ego motion--> current_ego_frame
new_camera_evidence ------------------> current_ego_frame
fused_grid = decay(previous_grid) + update(new_camera_evidence)
```

Implement simple log-odds occupancy updates:

```text
occupied += logit_hit
free     += logit_miss
unknown decays toward zero
```

This is classical, debuggable, and directly aligned with the roadmap.

### Phase 1D — Object Motion

For moving objects, use nuScenes 3D annotations as the reference truth. The dataset provides `instance_token` for cross-frame identity, so tracking is free.

Minimum state:

```text
instance_token
class_name
x, y, yaw
vx, vy
last_seen
```

Compute velocity from finite differences across consecutive samples (0.5s apart). Add a Kalman filter only if visualization shows the raw GT velocities are too jittery.

Use nuScenes annotations to validate the object layer first, then integrate a **public pretrained LiDAR 3D detector** rather than training our own detector from scratch.

Preferred detector path:

```text
pretrained CenterPoint / CenterPoint-style LiDAR detector
  -> predicted 3D boxes
  -> convert to our ego-frame Box3D format
  -> overlay predictions vs GT in BEV
  -> feed dynamic object layer
```

Training our own detector is deferred unless pretrained inference is blocked or we have a specific research reason to fine-tune.

### Methods Available

#### Classical / Geometric

Best for our immediate Phase 1.

- inverse perspective mapping for approximate road BEV
- depth back-projection from each camera
- calibrated multi-camera point splatting into ego frame
- log-odds occupancy mapping
- ego-motion grid warping
- Kalman filtering for object velocity
- map-prior overlay from nuScenes map expansion

Pros:
- debuggable
- works with one GPU
- teaches the system architecture
- does not require training

Cons:
- depth errors create noisy BEV
- camera-only free-space behind occluders is uncertain
- segmentation models trained outside nuScenes may be imperfect

#### Learning-Based 360 BEV

Use after the baseline exists. For 3D object detection, prefer **pretrained nuScenes weights first** rather than training from scratch.

- Lift-Splat-Shoot style view transformation
- BEVDet / BEVDepth / BEVStereo family for camera-only 3D detection
- BEVFormer for temporal camera-only BEV features
- BEVFusion for camera+LiDAR or camera-only BEV baselines
- CenterPoint / PointPillars for LiDAR-only 3D object detection
- MapTR / MapTRv2 for vectorized lane and road-element maps
- SurroundOcc / Occ3D / SparseOcc for semantic occupancy prediction

Pros:
- closer to modern research systems
- can produce cleaner BEV/object/occupancy outputs
- benchmarkable on nuScenes

Cons:
- heavy environment setup, often older PyTorch/MMDetection stacks
- training usually assumes multiple high-end GPUs
- harder to debug than a geometric baseline

### Model Landscape

Good candidates to study or integrate later:

| Model | Main Output | Why It Matters | Use Now? |
|---|---|---|---|
| CenterPoint | LiDAR 3D boxes, velocity, tracking-friendly object centers | best immediate fit for our LiDAR BEV + nuScenes annotations | integrate pretrained first |
| PointPillars | LiDAR pillar/BEV 3D detection | simpler classic LiDAR baseline, but weaker/older than CenterPoint | reference/fallback |
| BEVFormer | camera-only temporal BEV, 3D detection, map segmentation | directly matches multi-camera + temporal fusion | benchmark/reference |
| BEVDepth / BEVStereo | camera-only BEV 3D detection with explicit depth | conceptually close to our current depth pipeline | later |
| BEVFusion | unified BEV for camera/LiDAR detection and map segmentation | strong baseline, but repo environment is older/heavier | later/reference |
| MapTR / MapTRv2 | vectorized HD map elements, lanes, boundaries, centerlines | best fit for Phase 1 lane topology | after occupancy baseline |
| Occ3D | occupancy benchmark/labels for nuScenes | useful evaluation target for occupancy | use labels later |
| SurroundOcc | multi-camera 3D semantic occupancy | good research baseline for occupancy prediction | later |
| SparseOcc | efficient sparse 3D occupancy | modern occupancy direction | later |

### Concrete Phase 1 Checklist

- [x] Add initial 360 utilities in `fsd/`.
- [x] Add config/env support for `NUSCENES_ROOT=D:/nuscenes`.
- [x] Add a nuScenes metadata loader using direct JSON reads from the dataset root.
- [x] Add a `SurroundFrame` dataclass for six synchronized cameras.
- [x] Add a sample visualizer: six-camera contact sheet + ego pose info.
- [x] Add calibration validation code: project `LIDAR_TOP` points into camera images.
- [x] Run calibration validation smoke test when `D:/nuscenes` is mounted in the shell.
- [x] Add running 360 visualization: save/stream six-camera and LiDAR-overlay scene sequences.
- [x] Add smooth camera-sweep visualization using intermediate nuScenes camera frames.
- [x] Add ego-frame LiDAR BEV rasterizer and video CLI.
- [x] Render LiDAR BEV scene video when `D:/nuscenes` is mounted in the shell.
- [x] Add per-camera depth/segmentation inference wrappers that reuse existing modules where possible.
      Pivoted: kept front-camera drivable-area only (`fsd/inference.py::DrivableAreaPipeline`). Dropped monocular depth and 2D YOLO from the surround pipeline — see Phase 1B rationale.
- [x] Overlay nuScenes 3D annotation boxes on the LiDAR BEV (object layer).
- [ ] Formalize LiDAR BEV tensor channels: density, max/min/mean height, height range.
- [ ] Install/setup external MMDetection3D or OpenPCDet PointPillars runtime with nuScenes pretrained weights.
      Status: repo-side bridge is implemented in `fsd/pointpillars.py infer`, but this Windows Python 3.12 + Torch 2.7 + CUDA 12.8 environment does not currently have compatible compiled OpenMMLab `mmcv/mmdet/mmdet3d` ops installed.
- [x] Add PointPillars inference export command that reads `LIDAR_TOP` directly from `D:/nuscenes` and writes visualizer-ready ego-frame prediction JSON.
- [x] Add prediction JSON adapter for PointPillars/MMDetection3D outputs.
- [x] Visualize predicted 3D boxes vs GT boxes in BEV.
- [x] Project GT/predicted 3D boxes back onto the six camera images.
- [ ] Add temporal grid warping using nuScenes ego poses.
- [ ] Add log-odds occupancy fusion and decay over the LiDAR BEV.
- [ ] Add BEV video export for a short scene with the fused occupancy.
- [ ] Add ego-frame object velocity from annotation finite differences.
- [ ] Add evaluation hooks against nuScenes annotations/Occ3D labels later.

### First Implementation Target — Completed

Completed first code task:

> Create a read-only nuScenes scene loader and render a six-camera contact sheet for the first scene, with camera names, timestamps, and ego pose metadata.

That is the right first step because every later BEV/occupancy method depends on synchronized camera access and correct calibration.

Smoke-test output:

```text
outputs/nuscenes_contact_sheet_scene0_sample0.jpg
```

Next target:

> Project `LIDAR_TOP` points into the six camera images to validate calibration and coordinate transforms before BEV rasterization.

Implementation added:

```text
fsd/lidar_projection.py
```

Run when `D:/nuscenes` is mounted:

```powershell
.\.venv\Scripts\python.exe -m fsd.lidar_projection --dataroot D:/nuscenes --scene-index 0 --sample-index 0 --tile-width 360 --output outputs/nuscenes_lidar_projection_scene0_sample0.jpg
```

Smoke-test output:

```text
outputs/nuscenes_lidar_projection_scene0_sample0.jpg
```

Projected point counts for scene 0 / sample 0:

```text
raw LiDAR points: 34720
CAM_FRONT_LEFT: 3263
CAM_FRONT: 3364
CAM_FRONT_RIGHT: 2770
CAM_BACK_LEFT: 4055
CAM_BACK: 4318
CAM_BACK_RIGHT: 2998
```

Running visualization commands:

```powershell
.\.venv\Scripts\python.exe -m fsd.visualize --dataroot D:/nuscenes --scene-index 0 --frames 5 --view cameras --save --tile-width 360 --fps 2 --output outputs/nuscenes_cameras_sequence_smoke.mp4

.\.venv\Scripts\python.exe -m fsd.visualize --dataroot D:/nuscenes --scene-index 0 --frames 5 --view lidar --save --tile-width 360 --fps 2 --output outputs/nuscenes_lidar_sequence_smoke.mp4
```

Use `--stream` instead of `--save` to display the sequence in an OpenCV window and press `q` to quit.

Smooth camera sweep visualization:

```powershell
.\.venv\Scripts\python.exe -m fsd.visualize --dataroot D:/nuscenes --scene-index 0 --frames 233 --view cameras --sequence sweeps --save --tile-width 360 --fps 12 --output outputs/nuscenes_scene0_camera_sweeps_12fps.mp4
```

Output:

```text
outputs/nuscenes_scene0_camera_sweeps_12fps.mp4
233 frames, 12 FPS, 1080x480, 19.4 seconds
```

LiDAR BEV visualization:

```powershell
.\.venv\Scripts\python.exe -m fsd.visualize --dataroot D:/nuscenes --scene-index 0 --frames 40 --view bev --sequence keyframes --save --fps 2 --bev-resolution 0.25 --bev-scale 2 --output outputs/nuscenes_scene0_40f_bev_unified.mp4
```

Output:

```text
outputs/nuscenes_scene0_40f_bev_unified.mp4
40 frames, 2 FPS, 800x916, 20.0 seconds
```

`fsd.visualize` is the canonical visualization entry point:

```text
--view cameras   six-camera contact sheet
--view lidar     LiDAR projected into camera images
--view bev       ego-frame LiDAR BEV
--view gt_bev    ego-frame LiDAR BEV with nuScenes GT 3D boxes
```

3D object detection label validation:

```powershell
.\.venv\Scripts\python.exe -m fsd.visualize --dataroot D:/nuscenes --scenes 0 --frames 40 --view gt_bev --sequence keyframes --save --fps 2 --bev-resolution 0.25 --bev-scale 2 --output outputs/nuscenes_gt_boxes_bev_unified_40f.mp4
```

Output:

```text
outputs/nuscenes_gt_boxes_bev_unified_40f.mp4
40 frames, 2 FPS, 800x916, 20.0 seconds
```

### Phase 1E — Camera-only LSS BEV Vehicle Segmentation

Integrated NVIDIA's Lift-Splat-Shoot (ECCV 2020) BEV vehicle-segmentation model as a learned camera-only counterpart to the LiDAR BEV. This is the first learning-based BEV head in the project.

- Model + helpers ported into [fsd/lss.py](fsd/lss.py) (1:1 port of `LiftSplatShoot`, `CamEncode`, `BevEncode`, `Up`, `QuickCumsum`, and grid-config helpers from the original repo). The original repo is cloned to `external/lift-splat-shoot/` for reference and is git-ignored.
- Inference wrapper `LSSInference` consumes our `SurroundFrame` directly — no `nuscenes-devkit` runtime dependency. It builds the six per-camera intrinsics, extrinsics (calibrated_sensor rotation/translation), and the eval-time image transform (resize + center crop to 128x352) used in `src/data.py`.
- New `fsd.visualize` views: `lss_bev` (LSS vehicle-seg BEV, 200x200 @ 0.5 m/cell, same orientation as our LiDAR BEV — ego at center, forward = top) and `lss_lidar_bev` (LSS mask overlaid on the LiDAR BEV).
- Single dependency added: `efficientnet_pytorch==0.7.0`. EfficientNet ImageNet weights are not downloaded — `from_name` is used and the LSS checkpoint overwrites them.

Pretrained weights link (NVIDIA, ECCV 2020 release):
https://drive.google.com/file/d/1bsUYveW_eOqa4lglryyGQNeC4fyQWvQQ/view?usp=sharing

Place the checkpoint anywhere and pass it via `--lss-weights`. Suggested path: `models/lss_model_525000.pt` (the `models/` folder is git-ignored).

Run:

```powershell
.\.venv\Scripts\python.exe -m fsd.visualize --dataroot D:/nuscenes --scenes 0 --frames 40 --view lss_bev --sequence keyframes --save --fps 2 --bev-scale 2 --lss-weights models/lss_model_525000.pt --output outputs/nuscenes_scene0_40f_lss_bev.mp4

.\.venv\Scripts\python.exe -m fsd.visualize --dataroot D:/nuscenes --scenes 0 --frames 40 --view lss_lidar_bev --sequence keyframes --save --fps 2 --bev-scale 2 --lss-weights models/lss_model_525000.pt --output outputs/nuscenes_scene0_40f_lss_lidar_bev.mp4
```

Open items:

- [x] Smoke-test `lss_bev` and `lss_lidar_bev` end-to-end once the LSS checkpoint is on disk. Verify the BEV orientation matches the LiDAR BEV (forward = top, left = left).
- [ ] Consider adding a four-up panel view (six cameras + LSS BEV + LiDAR BEV + GT boxes) to mirror the LSS paper teaser figure.

Findings + fixes after first render pass:

- The standalone `lss_bev` view now renders the LSS probability map in a `cmap='Blues'`-style "paper" look (white background, deep-blue at high probability) with the metric grid and ego marker, upsampled 4x from the native 200x200 to ~800x800 for a smooth read. A `style="dark"` mode is also available for parity with the LiDAR BEV's dark canvas.
- The original `lss_lidar_bev` overlay was shifted by ~2x because the LSS grid (200x200 @ 0.5 m/cell) was pasted onto the LiDAR BEV (400x400 @ 0.25 m/cell, then scale=2 -> 800x800) without resampling to the correct pixel size, and the header height was not scaled along with the LiDAR canvas. Fixed by passing `lidar_resolution` and `lidar_scale` into `overlay_lss_on_lidar_bev` and resizing the LSS grid to the actual LiDAR-canvas pixel dims.

About the original LSS paper's map background:

- The map polygons behind LSS predictions in the NVIDIA repo come from the nuScenes **map_expansion** vector-map archive (loaded via `NuScenesMap` in `nuscenes-devkit`), not from any per-frame model output.
- The expansion archive has now been extracted to `D:/nuscenes/maps/expansion/` (boston-seaport.json plus three Singapore JSONs). nuScenes data itself is mixed: 467 Boston scenes + 383 Singapore scenes across 3 districts. Scene -> location is looked up per-scene from `log.json`.
- We parse the expansion JSON directly (kept the no-`nuscenes-devkit` runtime rule) in [fsd/nuscenes_map.py](fsd/nuscenes_map.py). Layers + colors mirror the original LSS visualization (`road_segment` + `lane` fill, `road_divider` + `lane_divider` lines, plus a light `ped_crossing` overlay). `NuScenesMapRenderer.render()` returns a 2D ego-frame canvas at any requested pixel size.
- `render_lss_bev` accepts an optional `map_background`. `fsd.visualize` auto-enables the map background for LSS views when the expansion folder is present; disable with `--no-map`.

About the "Shoot" head of LSS:

- "Shoot" in the LSS paper refers to **template trajectory shooting**: candidate ego trajectories are scored against a predicted BEV cost map, and the lowest-cost one is chosen for planning (Section 4.4 of the ECCV 2020 paper).
- The released open-source repo only ships the **lift + splat vehicle-segmentation head**. The cost-map head, the trajectory templates, and the shooting code are **not** in the upstream repo and therefore not in our port. Adding "shoot" would require training (or hand-defining) a cost head and authoring the trajectory-shooting code from the paper description.
