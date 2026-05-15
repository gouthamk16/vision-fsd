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

Use the current project strengths first:

- current YOLO vehicle detection/tracking
- current Depth Anything metric depth
- current drivable segmentation concept
- nuScenes camera intrinsics/extrinsics
- nuScenes ego pose instead of our fragile SIFT odometry

For each camera:

1. Load image directly from `D:/nuscenes`.
2. Run detection/segmentation/depth.
3. Back-project useful pixels or object centers into that camera's 3D frame.
4. Transform camera points into ego frame using calibrated sensor extrinsics.
5. Rasterize into a shared BEV grid.

First BEV grid:

```text
range: x=[-50m, 50m], y=[-50m, 50m]
resolution: 0.25m or 0.5m per cell
channels:
  occupied_prob
  free_prob
  drivable_prob
  dynamic_object_prob
  semantic_class_id
  last_observed_time
```

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

For moving objects, start with nuScenes 3D annotations as the reference truth, then replace with model predictions later.

Minimum state:

```text
track_id
class_name
x, y, yaw
vx, vy
last_seen
```

Start with finite differences over annotation boxes or detected BEV centroids. Add a Kalman filter after the first visualization works.

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

Use after the baseline exists.

- Lift-Splat-Shoot style view transformation
- BEVDet / BEVDepth / BEVStereo family for camera-only 3D detection
- BEVFormer for temporal camera-only BEV features
- BEVFusion for camera+LiDAR or camera-only BEV baselines
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
- [ ] Add calibration validation: project `LIDAR_TOP` points into camera images.
- [ ] Add per-camera depth/segmentation inference wrappers that reuse existing modules where possible.
- [ ] Add camera-to-ego back-projection.
- [ ] Add single-frame 360 BEV rasterization.
- [ ] Add temporal grid warping using nuScenes ego poses.
- [ ] Add log-odds occupancy fusion and decay.
- [ ] Add BEV video export for a short scene.
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
