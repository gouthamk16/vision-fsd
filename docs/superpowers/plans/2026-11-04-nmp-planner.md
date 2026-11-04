# Neural Motion Planner (NMP) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train NMP (Zeng et al., CVPR 2019, arXiv 2101.06679) on our nuScenes subset: a LiDAR + HD-map conv net that detects vehicles, forecasts their boxes 3 s ahead, and outputs a space-time cost volume that scores sampled ego trajectories. Show it in `fsd.visualize` as `planner_bev`.

**Architecture:** A new `fsd/nmp/` package. An offline step turns each keyframe into a compact cached sample: occupied-voxel indices for 10 sweeps, map masks, the ego future, and GT boxes now and at 6 future keyframes. A PIXOR-style backbone feeds a dense detection/forecast head at stride 4 and a cost-volume head at full resolution. Training uses perception loss plus the paper's max-margin planning loss against sampled trajectories. Inference samples curvature-linear trajectories (straight / circle / clothoid) and picks the minimum summed cost.

**Tech Stack:** Python 3.12, torch 2.7 cu128 (bf16 autocast), numpy, opencv. No new dependencies.

**Spec:** `knowledge/summary_neural_motion_planner.md` (paper summary + nuScenes mapping) and the dataset decision in this conversation: nuScenes trainval, blobs 01–02 extracted at `C:/nuscenes` (170 scenes: 139 official-train, 31 official-val).

## Global Constraints

- Grid: x (forward) in [-70.4, 70.4] m, y (left) in [-40, 40] m, 0.2 m cells, so 704 x 400. z in [-2.0, 3.4] m at 0.2 m (27 bins). This is the paper's ROI.
- Input: 10 sweeps at stride 2 over the 20 Hz LIDAR_TOP stream, so 1.0 s of history, the paper's 10 Hz x 10. One binary occupancy channel per (sweep, z-bin), plus 8 map channels: 10*27 + 8 = 278 channels.
- Horizon: T = 6 future steps. Step k is the k-th next keyframe (nominal t = 0.5 k s, so 0.5 ... 3.0 s). The ego future is those keyframes' LIDAR_TOP ego poses. Future boxes come from those keyframes' annotations.
- Map layers, in channel order: `drivable_area, lane, ped_crossing, walkway, stop_line, carpark_area, road_divider, lane_divider`.
- Detected classes (the paper detects vehicles): `car, truck, bus, trailer, construction`. Collision checks use every class in `object_detection.DETECTION_CLASSES`.
- Sampler: P(straight) = 0.5, P(circle) = 0.25, P(clothoid) = 0.25. Clothoid scale a ~ U[6, 80] m, curvature rate c = ±π/a². Acceleration ~ U[-5, 5] m/s². Speed is clamped at >= 0 (no reversing). |curvature| <= 0.2 1/m. Circle curvature ~ U[-0.2, 0.2]. 1000 trajectories per sample, in training and at inference.
- Training negatives ignore the true initial state with probability 0.8 (paper). They then use v0 ~ U[0, 15] m/s and k0 ~ U[-0.2, 0.2].
- Cost volume clipped to [-1000, 1000]. Margin violation constants: collision 200, leaving drivable area 100.
- Loss = detection + β · planning, with β = 0.01. Planning loss is per sample: max over negatives of Σ_t ReLU(ĉ_t − c_i^t + d_i^t + γ_i^t), averaged over the batch.
- Split: val = the official nuScenes `val` scene list; train = every other available scene.
- The ego state (v0, k0) feeds only the sampler, never the network. This keeps the "ego status shortcut" out of the learned cost.
- Deliberate deviations from the paper, each kept because it changes nothing the paper's results depend on:
  - **Detection head:** anchor-free PIXOR-style (per-cell classification and regression) with focal loss, instead of 12 SSD anchors with IoU matching and hard-negative mining. The paper's contribution is the cost volume, and the backbone is already PIXOR's.
  - **Clothoids:** integrated numerically as curvature-linear curves instead of with Fresnel integrals. It is the same curve family, without a scipy dependency.
  - **No red-light term in γ:** nuScenes has no traffic-light state.
  - **Off-road term in γ:** added, because the map makes it available.
- Code standards per CLAUDE.md: files <= ~300 lines, functions <= ~30 lines, typed dataclasses across module boundaries.
- `fsd/nmp/train.py` is the second documented CLI exception after `centerpoint_export.py`, because training is not a view. Update CLAUDE.md and AGENTS.md in Task 9.

## Review Focus

1. **Keyframes near a scene's start or end** (fewer than 19 sweeps of history, or fewer than 6 future keyframes): skipped when the cache is built, never crash and never zero-padded. Test in Task 3.
2. **Stationary ego** (v0 = 0, including a zero-length pose delta when estimating curvature): k0 = 0, no NaN, and trajectories with negative acceleration stay at the origin. Tests in Tasks 2 and 3.
3. **Trajectory points outside the 140.8 x 80 m grid** (fast negatives, a sideways circle): cost lookup and violation lookup clamp to the edge cell, with no index error. Test in Task 6.
4. **An object missing from a future keyframe** (it left the scene or was occluded): masked out of the regression targets, the collision raster and the metrics for that step only. Tests in Tasks 3 and 6.
5. **A frame with no vehicles at all:** detection loss is finite, with focal loss normalised by max(1, positives). Test in Task 6.

---

### Task 1: Per-layer map masks

**Files:**
- Modify: `fsd/nuscenes_map.py`
- Test: `tests/test_nuscenes_map.py`

**Interfaces:**
- Produces: `MAP_LAYERS: tuple[str, ...]` (the 8 names above, in order) and `NuScenesMapRenderer.layer_masks(frame: SurroundFrame, grid: BevGrid, layers: tuple[str, ...] = MAP_LAYERS) -> np.ndarray` of shape `(len(layers), H, W)`, dtype uint8, values in {0, 1}.

- [ ] **Step 1: Write the failing test** `test_layer_masks_rasterise_each_layer_in_its_own_channel`. Reuse `_write_map` from the culling test, extended with a `lane` polygon and a `lane_divider` line. Assert `masks.shape == (8, *grid.shape)`. The drivable cell at the square's centre must be 1 in channel 0 only. A cell under the divider must be 1 in channel 7.
- [ ] **Step 2:** Run `pytest tests/test_nuscenes_map.py -v`. Expected: FAIL, `layer_masks` not defined.
- [ ] **Step 3:** Implement it. Extend `_load` with the polygon layers `lane`, `stop_line` and `carpark_area`; each record has `polygon_token`. Reuse the existing culling (`visible`). Fill polygons with `cv2.fillPoly(..., 1)`. Draw lines 2 px wide. Leave `draw`/`render` output unchanged.
- [ ] **Step 4:** Run `pytest tests/test_nuscenes_map.py -v`. Expected: PASS (2 tests).
- [ ] **Step 5:** Commit: `feat: rasterise nuScenes map layers to per-layer BEV masks`

### Task 2: Config and trajectory sampler

**Files:**
- Create: `fsd/nmp/__init__.py` (empty), `fsd/nmp/config.py`, `fsd/nmp/trajectories.py`
- Test: `tests/test_nmp_trajectories.py`

**Interfaces:**
- Produces: `NmpConfig` (frozen dataclass). Defaults are the Global Constraints values:
  - `x_range, y_range, resolution, z_range, z_resolution, n_sweeps=10, sweep_stride=2, horizon=6, dt=0.5, map_layers, det_classes, n_trajectories=1000, accel_range=(-5.0, 5.0), clothoid_a_range=(6.0, 80.0), max_curvature=0.2, p_straight=0.5, p_circle=0.25, negative_free_init_p=0.8, init_speed_range=(0.0, 15.0), gamma_collision=200.0, gamma_offroad=100.0, cost_clip=1000.0, beta=0.01`
  - Properties: `grid -> BevGrid`, `z_bins -> int` (27), `in_channels -> int` (278), `times -> np.ndarray` ([0.5, 1.0, ..., 3.0]), `history_sweeps -> int` ((n_sweeps − 1) · stride + 1 = 19).
- Produces: `rollout(v0, k0, rate, accel, times, substeps=10) -> Tensor[..., T, 3]`. Leading dims broadcast. Output is (x, y, yaw) in the start's ego frame. Integrate heading over arc length ξ with κ(ξ) = clip(k0 + rate·ξ, ±max_curvature), v(t) = max(0, v0 + accel·t), using midpoint steps of `dt / substeps`.
- Produces: `sample_trajectories(v0: Tensor[B], k0: Tensor[B], cfg: NmpConfig, n: int, obey_initial_p: float = 1.0, generator: torch.Generator | None = None) -> Tensor[B, n, T, 3]`.
  - Each trajectory first picks its type.
  - Straight: k0 := 0, rate := 0.
  - Circle: k0 ~ U(±max_curvature), rate := 0.
  - Clothoid: k0 := the initial curvature, rate := ±π/a².
  - With probability `1 − obey_initial_p`, its (v0, k0) are redrawn from `init_speed_range` / ±max_curvature.

- [ ] **Step 1: Write failing tests**
  - `test_straight_constant_speed_moves_along_x`: `rollout(10., 0., 0., 0., times)` gives x ≈ 10·t, y ≈ 0, yaw ≈ 0 (atol 1e-3).
  - `test_circle_stays_on_its_radius`: rollout with k0 = 0.1 and rate 0 gives points at distance 10 m from (0, 10), atol 0.05.
  - `test_braking_never_reverses`: v0 = 2, accel = −5 gives x non-decreasing, and the final x ≈ 0.4 (= v0²/2a).
  - `test_stationary_ego_with_negative_accel_stays_put`: v0 = 0, accel = −3 gives all-zero positions.
  - `test_sampler_shapes_and_initial_heading`: shape `(2, 1000, 6, 3)`. With `obey_initial_p = 1`, every clothoid's initial heading rate ≈ k0. Check yaw of the first step ≈ k0 · ξ1 for small ξ1.
- [ ] **Step 2:** Run `pytest tests/test_nmp_trajectories.py -v`. Expected: FAIL (import error).
- [ ] **Step 3:** Implement `config.py` and `trajectories.py` with the signatures above. Use vectorised torch only, with no Python loop over trajectories.
- [ ] **Step 4:** Run `pytest tests/test_nmp_trajectories.py -v`. Expected: PASS (5).
- [ ] **Step 5:** Commit: `feat(nmp): config and curvature-linear trajectory sampler`

### Task 3: Training samples and the cache

**Files:**
- Create: `fsd/nmp/samples.py`, `fsd/nmp/val_scenes.txt`, `fsd/nmp/train.py` (`cache` subcommand only)
- Test: `tests/test_nmp_samples.py`

**Interfaces:**
- Consumes: `NuScenesSceneLoader.iter_scene_frames(scene_name=..., include_lidar=True, lidar_sweeps=cfg.history_sweeps)`, `lidar_points_ego`, `NuScenesAnnotationLoader.category_of_instance`, `box_from_global`, `NuScenesMapRenderer.layer_masks`, `Pose.relative_to`.
- Produces: `NmpSample` (dataclass, all arrays in the current ego frame):
  - `token: str`, `scene: str`
  - `voxels: np.ndarray[uint32]`: unique flat indices into `(n_sweeps * z_bins, H, W)`, channel = `sweep * z_bins + z_bin`
  - `map_masks: np.ndarray[uint8] (8, H, W)`
  - `ego_future: np.ndarray[float32] (T, 3)` (x, y, yaw)
  - `v0: float`, `k0: float`
  - `boxes: np.ndarray[float32] (K, T+1, 5)`: x, y, w, l, yaw, where step 0 is now
  - `box_valid: np.ndarray[bool] (K, T+1)`
  - `box_class: np.ndarray[int8] (K,)`: index into `tuple(sorted(set(DETECTION_CLASSES.values())))`
- Produces: `voxelize(points: np.ndarray, sweep: np.ndarray, cfg) -> np.ndarray[uint32]`, `ego_state(lidar: LidarFrame) -> tuple[float, float]`, `build_scene_samples(loader, map_renderer, annotations: NuScenesAnnotationLoader, scene_name: str, cfg) -> list[NmpSample]`, `save_sample(sample, path)`, `load_sample(path) -> NmpSample` (compressed npz), and `split_scenes(loader) -> dict[str, list[str]]` with keys `train` and `val`.
- `ego_state`:
  - v0 = |Δxy| / Δt between `sweeps[0]` and `sweeps[2]` (0.1 s apart).
  - k0 = Δyaw / Δξ when Δξ > 0.5 m, else 0.
- Sweep selection: `lidar.sweeps[::cfg.sweep_stride][:cfg.n_sweeps]`.
- A keyframe is skipped (no sample) when it has fewer than `history_sweeps` sweeps or fewer than T later keyframes in its scene.
- `val_scenes.txt`: the 150 names of `val` from nuscenes-devkit `python-sdk/nuscenes/utils/splits.py`, one per line, sorted. Use the master copy at https://raw.githubusercontent.com/nutonomy/nuscenes-devkit/master/python-sdk/nuscenes/utils/splits.py.
- CLI: `python -m fsd.nmp.train cache --split {train,val} [--cache-dir ~/.cache/vision-fsd/nmp]` writes `<cache-dir>/<split>/<token>.npz` and prints the sample count.

- [ ] **Step 1: Write failing tests** (pure, no dataset needed)
  - `test_voxelize_indexes_sweep_z_row_col`: a point at (70.3, 39.9, −1.9) in sweep 1 maps to flat index `((1*27 + 0) * 704 + 0) * 400 + 0`. A point outside the grid or the z-range is dropped. Duplicates collapse to one index.
  - `test_ego_state_stationary_is_zero_curvature`: identical sweep poses give `(0.0, 0.0)`.
  - `test_ego_state_turning`: poses 1 m apart with Δyaw 0.05 rad over 0.1 s give v0 ≈ 10 and k0 ≈ 0.05.
  - `test_box_missing_in_future_is_masked`: build the future-box table for an instance present at steps 0–2 only. Steps 3–6 must have `box_valid` False. Call `_future_boxes(frames: list[SurroundFrame], annotations: list[list[dict]], category_of_instance: dict[str, str]) -> tuple[boxes, box_valid, box_class]`. Here `frames[0]` is the current keyframe, `annotations[k]` holds the records of `frames[k]`, and rows are the union of instances present at any step. Detection targets use only rows valid at step 0; collisions use every valid (row, step). Also assert that an instance first seen at step 4 has `box_valid` False for steps 0–3. Feed it synthetic `SurroundFrame`s from `conftest.make_frame`.
  - `test_save_load_round_trip`: every array is equal after the npz round trip.
- [ ] **Step 2:** Run `pytest tests/test_nmp_samples.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `samples.py`, `val_scenes.txt` and the `cache` subcommand.
- [ ] **Step 4:** Run `pytest tests/test_nmp_samples.py -v`. Expected: PASS (5).
- [ ] **Step 5: Build the real cache**
  - Run `python -m fsd.nmp.train cache --split val`, then `--split train`.
  - Expected counts: val ≈ 31 × 26 ≈ 800, train ≈ 139 × 26 ≈ 3.6k (40 keyframes minus ~4 without history, minus 6 without future, per scene).
  - Expected size: < 1 MB per sample on average.
  - Record the exact counts in the commit message.
- [ ] **Step 6:** Commit: `feat(nmp): build cached training samples from nuScenes keyframes`

### Task 4: Dataset

**Files:**
- Create: `fsd/nmp/dataset.py`
- Test: `tests/test_nmp_dataset.py`

**Interfaces:**
- Consumes: `load_sample`, `NmpConfig`, `ego_motion.EGO_FOOTPRINT`, `object_detection.Box3D.footprint` (build a `Box3D` to reuse its corner order).
- Produces: `NmpDataset(cache_dir: Path, split: str, cfg)`. `__getitem__` returns a dict of tensors:
  - `input`: uint8 (278, 704, 400), densified voxels plus map masks
  - `ego_future`: (T, 3)
  - `v0`, `k0`: scalars
  - `boxes`: (K, T+1, 5), `box_valid`: (K, T+1), `box_class`: (K,)
  - `obstacles`: bool (T, 704, 400). Every valid box at step k (all classes) rasterised and dilated by 1.0 m, which approximates the ego half-width.
  - `drivable`: bool (704, 400), map channel 0
- Produces: `collate(items) -> dict` that pads `boxes` / `box_valid` / `box_class` to the batch's max K with `box_valid = False`.

- [ ] **Step 1: Write failing tests** on a temp cache written with `save_sample` from a synthetic `NmpSample`:
  - `test_densify_matches_voxel_indices`: `input.sum() == len(voxels) + map_masks.sum()`, and one known voxel is 1.
  - `test_obstacles_only_where_box_valid`: a box valid at step 1 but not step 2 appears in `obstacles[0]` (k = 1 ↔ index 0) and not in `obstacles[1]`.
  - `test_collate_pads_boxes`: samples with K = 2 and K = 0 collate to `(2, 2, T+1, 5)`, and `box_valid[1]` is all False.
- [ ] **Step 2:** Run `pytest tests/test_nmp_dataset.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `dataset.py`. Rasterise boxes with `cv2.fillPoly` on the config grid, then dilate with an elliptical kernel of radius 5 cells.
- [ ] **Step 4:** Run `pytest tests/test_nmp_dataset.py -v`. Expected: PASS (3).
- [ ] **Step 5:** Commit: `feat(nmp): dataset densifying cached samples`

### Task 5: Model

**Files:**
- Create: `fsd/nmp/model.py`
- Test: `tests/test_nmp_model.py`

**Interfaces:**
- Produces: `NmpOutput` (dataclass):
  - `cls_logits`: (B, 1, H/4, W/4)
  - `box_reg`: (B, (T+1)·6, H/4, W/4). Per step: dx, dy, log w, log l, sin, cos.
  - `cost`: (B, T, H, W), clipped to ±`cfg.cost_clip`
- Produces: `NeuralMotionPlanner(cfg)` with `forward(x: Tensor[B, C, H, W]) -> NmpOutput`.
- Backbone (paper, from PIXOR):
  - Five blocks of {2, 2, 3, 6, 5} 3×3 conv–BN–ReLU layers with {32, 64, 128, 256, 256} filters, max-pool after blocks 1–3.
  - Block 1–4 outputs are bilinearly resized to H/4 × W/4 and concatenated (480 channels) as the input to block 5.
- Detection head: two 3×3 conv layers (256) per branch, then 1×1 to 1 channel (cls) and to (T+1)·6 (reg). Initialise the cls bias to −log((1−0.01)/0.01).
- Cost head (paper): deconv 128 (3×3, stride 2) + conv 128, deconv 64 + conv 64, then a 1×1 conv to T. The output is at full H × W.
- The input is cast to float inside `forward`.

- [ ] **Step 1: Write failing tests**
  - `test_output_shapes_small_grid`: a config with an 80 × 48 grid and `in_channels = 278`, batch 2 on CPU, gives shapes `(2,1,20,12)`, `(2,42,20,12)`, `(2,6,80,48)`.
  - `test_cost_is_clipped`: after setting the final cost conv bias to 5000, `out.cost.max() <= 1000`.
- [ ] **Step 2:** Run `pytest tests/test_nmp_model.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `model.py`.
- [ ] **Step 4:** Run `pytest tests/test_nmp_model.py -v`. Expected: PASS (2).
- [ ] **Step 5: Check it fits the GPU.** Run a forward + backward at batch 4, full 704 × 400, under `torch.autocast("cuda", torch.bfloat16)`. Expected: peak `torch.cuda.max_memory_allocated()` < 7 GB. If it is over, use batch 2 and record that in the commit. If CUDA is unavailable, check the laptop GPU mode first (it must be Standard).
- [ ] **Step 6:** Commit: `feat(nmp): PIXOR backbone with detection and cost-volume heads`

### Task 6: Losses

**Files:**
- Create: `fsd/nmp/losses.py`
- Test: `tests/test_nmp_losses.py`

**Interfaces:**
- Consumes: `NmpOutput`, the batch dict from Task 4, `sample_trajectories`.
- Produces: `detection_targets(boxes, box_valid, box_class, cfg) -> tuple[cls (B,1,h,w), reg (B,(T+1)·6,h,w), reg_mask (B,T+1,h,w)]`.
  - Positive cells are stride-4 cell centres inside a valid step-0 box of a detected class.
  - dx, dy = (box centre at step k − cell centre) / 4.0 m.
  - log w and log l use metres.
- Produces: `detection_loss(out, targets) -> Tensor`. Sigmoid focal loss (α 0.25, γ 2) summed and divided by max(1, positives), plus smooth-L1 over masked regression divided by max(1, masked entries).
- Produces: `lookup_costs(cost, traj, grid) -> Tensor[B, N, T]`. Nearest cell, clamped to the grid.
- Produces: `violations(traj, obstacles, drivable, cfg) -> Tensor[B, N, T]`. Evaluated at the body centre: the trajectory point plus 1.385 m along yaw. Adds `gamma_collision` where `obstacles[t]` is set and `gamma_offroad` where `drivable` is unset, clamped to the grid.
- Produces: `planning_loss(cost, gt, negatives, gamma, grid) -> Tensor`. Margin d = ‖gt_xy − neg_xy‖ per step, then the paper's formula from Global Constraints.

- [ ] **Step 1: Write failing tests**
  - `test_planning_loss_zero_when_gt_cheapest_by_margin`: cost is 0 on the GT cells and 1000 elsewhere, giving loss 0.
  - `test_planning_loss_positive_when_negative_cheaper`: a single negative on cells with cost −1000 gives loss > 0. Its gradient on those cells is negative.
  - `test_collision_adds_gamma_only_at_that_step`: obstacles set only at step 2 under the negative's body centre give γ = 200 at step 2 and 0 elsewhere.
  - `test_out_of_grid_points_clamp`: traj x = 500 m does not raise and reads the edge cell.
  - `test_detection_loss_finite_without_vehicles`: all `box_valid` False gives a finite loss > 0 (background focal term).
  - `test_target_encoding_round_trip`: decoding `reg` at a positive cell recovers the box centre and yaw at steps 0 and 3 (atol 1e-4).
- [ ] **Step 2:** Run `pytest tests/test_nmp_losses.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `losses.py`.
- [ ] **Step 4:** Run `pytest tests/test_nmp_losses.py -v`. Expected: PASS (6).
- [ ] **Step 5:** Commit: `feat(nmp): detection and max-margin planning losses`

### Task 7: Planner inference, metrics and baseline

**Files:**
- Create: `fsd/nmp/planner.py`, `fsd/nmp/metrics.py`
- Test: `tests/test_nmp_planner.py`

**Interfaces:**
- Consumes: `NeuralMotionPlanner`, `sample_trajectories`, `lookup_costs`, `samples.voxelize`, `samples.ego_state`, `NuScenesMapRenderer.layer_masks`.
- Produces: `select_plan(cost: Tensor[T,H,W], candidates: Tensor[N,T,3], grid) -> tuple[int, Tensor[N]]`, returning the argmin index and the summed costs.
- Produces: `decode_detections(out: NmpOutput, cfg, threshold=0.3) -> list[list[Box3D]]` (per batch item) plus forecasts `(K, T+1, 3)`. Uses 3×3 max-pool peak picking.
- Produces: `PlanResult` (plan (T,3), candidates (N,T,3), costs (N,), cost_volume (T,H,W), detections, forecasts), and `NmpPlanner(weights: Path, cfg, device)` with `plan(frame, lidar, map_renderer) -> PlanResult`.
- Produces: `boxes_overlap(a: np.ndarray[4,2], b: np.ndarray[4,2]) -> bool` (separating-axis test), `ego_footprint(pose: np.ndarray[3]) -> np.ndarray[4,2]` (`EGO_FOOTPRINT` placed at x, y, yaw), `constant_velocity_plan(v0, cfg) -> np.ndarray[T,3]`, and `PlanMetrics` with `add(plan, ego_future, boxes, box_valid, drivable, cfg)` and `summary() -> dict`.
  - The summary has L2 at 1/2/3 s (steps 2/4/6), box-collision rate up to 1/2/3 s (cumulative: any overlap at or before that step), and off-drivable rate up to 1/2/3 s, all in %.

- [ ] **Step 1: Write failing tests**
  - `test_select_plan_picks_min_summed_cost`.
  - `test_boxes_overlap_rotated_cases`: touching, separated and 45°-rotated pairs.
  - `test_collision_rate_is_cumulative`: one collision at step 2 counts at 1 s, 2 s and 3 s, and 0 before.
  - `test_cv_plan_goes_straight_at_v0`.
  - `test_missing_box_step_never_collides`: a box overlapping the plan at a step where `box_valid` is False is not counted.
- [ ] **Step 2:** Run `pytest tests/test_nmp_planner.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `metrics.py` and `planner.py`.
- [ ] **Step 4:** Run `pytest tests/test_nmp_planner.py -v`. Expected: PASS (5).
- [ ] **Step 5:** Commit: `feat(nmp): planner inference, planning metrics and constant-velocity baseline`

### Task 8: Training and evaluation

**Files:**
- Modify: `fsd/nmp/train.py` (add `fit` and `eval`)
- Test: `tests/test_nmp_train.py`

**Interfaces:**
- Consumes: everything above.
- Produces the CLI:
  - `python -m fsd.nmp.train fit [--cache-dir] [--out checkpoints/nmp] [--epochs 20] [--batch-size 4] [--lr 2e-4] [--workers 4] [--limit N]`. Uses Adam with cosine decay to 0 under bf16 autocast. Each step draws 1000 negatives per sample with `obey_initial_p = 0.2`. Logs `det`, `plan` and `total` every 50 steps. After each epoch it computes val `PlanMetrics`, saves `last.pt`, and keeps `best.pt` by val collision rate at 3 s (ties broken by L2 at 3 s).
  - `python -m fsd.nmp.train eval --weights checkpoints/nmp/best.pt [--split val]`. Prints one markdown table: rows `CV baseline`, `NMP (ours)`, `NMP ref (Khurana et al. ECCV'22, nuScenes-val, no map)` with the reference values L2 0.53/1.25/2.67 m and box collision 0.04/0.12/0.87 %, and columns L2, collision and off-road at 1/2/3 s. Below the table it prints vehicle detection AP at 2 m centre distance and forecast L2 of matched detections at 1/2/3 s.

- [ ] **Step 1: Write the failing test** `test_fit_overfits_four_samples`. On a synthetic 4-sample cache with a small grid config, `fit(limit=4, epochs=30)` on CPU must cut the planning loss by ≥ 90 %, and the planned trajectory must match the GT within 1 m at 3 s.
- [ ] **Step 2:** Run `pytest tests/test_nmp_train.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `fit` and `eval`.
- [ ] **Step 4:** Run `pytest tests/ -v`. Expected: all pass.
- [ ] **Step 5: Smoke run on real data:** `python -m fsd.nmp.train fit --limit 64 --epochs 3`. Expected: no error, both loss terms finite and decreasing, about 0.3 s per step at batch 4.
- [ ] **Step 6: Full run:** `python -m fsd.nmp.train fit` (about 20 epochs; budget a few hours), then `python -m fsd.nmp.train eval --weights checkpoints/nmp/best.pt`. Acceptance:
  - (a) NMP collision at 3 s ≤ CV baseline collision at 3 s.
  - (b) NMP off-road at 3 s < CV off-road at 3 s.
  - (c) Vehicle AP at 2 m ≥ 0.4.
  - Paste the table into the commit message.
  - If (a) fails, note it. Possible follow-ups, for a separate decision:
    - More data: blobs 03–05 add about 255 scenes, at 31 GB each.
    - Planning-only samples from non-key sweeps.
    - Tuning β.
- [ ] **Step 7:** Commit: `feat(nmp): training and evaluation CLI`

### Task 9: `planner_bev` view and docs

**Files:**
- Modify: `fsd/visualize.py`, `CLAUDE.md`, `AGENTS.md`, `README.md`
- Test: `tests/test_visualize.py`

**Interfaces:**
- Consumes: `NmpPlanner.plan`, `bev.lidar_canvas`, `draw_boxes_on_bev`, `bev.draw_polygon`.
- Produces: the view `planner_bev`, which requires `--nmp-weights`.
  - When it is selected, the loader fetches `max(--lidar-sweeps, cfg.history_sweeps)` sweeps. `_Products.points` keeps rendering only the first `--lidar-sweeps` of them (`LidarFrame(lidar.sweeps[:n])`).
  - Drawn layers:
    - LiDAR canvas over the dark map
    - Cost volume: per-cell minimum over t, normalised and blended with `COLORMAP_INFERNO` at alpha 0.5
    - 100 of the candidates, in gray
    - The selected plan as a green polyline with dots at each 0.5 s step
    - Detections with forecast centre dots
    - Header: `v0` and the plan's summed cost
  - `all` includes it only when `--nmp-weights` is given (same rule as LSS).

- [ ] **Step 1: Write the failing test** `test_all_skips_planner_without_weights`. Extends the existing selection test: `_select_views("all", None, None)` excludes `planner_bev`, and asking for `planner_bev` without weights raises `ValueError` matching `--nmp-weights`. Update `_select_views` to take `(view, lss_weights, nmp_weights)`.
- [ ] **Step 2:** Run `pytest tests/test_visualize.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement the view.
- [ ] **Step 4:** Run `pytest tests/ -v`. Expected: all pass. Then run `python -m fsd.visualize --view planner_bev --nmp-weights checkpoints/nmp/best.pt --scene-name <a val scene> --frames 20 --save` and inspect frames. Expected: the plan follows the lane, the cost is low on the drivable area, and the cost is high on parked cars.
- [ ] **Step 5: Update the docs.**
  - CLAUDE.md and AGENTS.md:
    - Add `fsd/nmp/` to the architecture section (it consumes `data`, `lidar_projection`, `nuscenes_map`, `object_detection`; nothing imports it except `visualize`).
    - Document the `fsd.nmp.train` CLI exception.
    - Replace "There is no motion planner at the moment" with how to build the cache, train and evaluate.
  - README: add an NMP section with the eval table, one `planner_bev` image and the commands.
- [ ] **Step 6:** Commit: `feat: planner_bev view for the neural motion planner` and `docs: document the NMP planner`
