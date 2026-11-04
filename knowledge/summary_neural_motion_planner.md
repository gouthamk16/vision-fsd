# NMP: End-to-end Interpretable Neural Motion Planner

Zeng, Luo, Suo, Sadat, Yang, Casas, Urtasun (Uber ATG / U. Toronto), CVPR 2019.
arXiv: [2101.06679](https://arxiv.org/abs/2101.06679)

## The idea in one line

A single conv net reads LiDAR sweeps and a rasterized HD map. It outputs (a) 3D detections with
their future boxes and (b) a **space-time cost volume** $c \in \mathbb{R}^{T \times H \times W}$.
Planning is: sample many physically feasible trajectories, look up each one's cost in the volume,
and take the cheapest.

$$
\mathbf{s}^* = \arg\min_{\mathbf{s}} \sum_{t=0}^{T-1} c^t(\mathbf{s}^t)
$$

The cost volume is never supervised directly. A max-margin loss pushes the human trajectory to
be the cheapest. The detection and forecasting heads are there for interpretability and to give
the shared features something dense to learn from (planning-only training does worse, see the
ablation).

## Architecture

**Input.** $T' = 10$ past sweeps, ego-motion compensated into the current ego frame, voxelized
to binary occupancy. Time is folded into channels, so the net uses only 2D convs:
$H \times W \times (Z T' + M)$, where $M$ is the number of map raster channels (road,
intersections, lanes, lane boundaries, traffic lights, ...).
Paper ROI: $\pm 70.4$ m longitudinal, $\pm 40$ m lateral, $z \in [-2, 3.4]$ m, so a
$704 \times 400 \times 27$ grid at 0.2 m.

**Backbone.** PIXOR-style: 5 blocks with $\lbrace 2,2,3,6,5 \rbrace$ 3x3 conv layers and
$\lbrace 32,64,128,256,256 \rbrace$ filters, max-pool after blocks 1-3. The outputs of blocks 1-4 are resized
to 1/4 resolution and concatenated before block 5, which gives multi-scale features. Total stride 4.

**Perception head.** SSD-style, 12 anchors per cell. It outputs a classification score $p$ and,
**for every future timestep** $t = 0..T-1$, the regression targets
$(l_x^t, l_y^t, s_w^t, s_h^t, a_{\sin}^t, a_{\cos}^t)$. Forecasting is just regression at every $t$.

**Cost volume head.** Two deconvs (128, 64 filters, stride 2, each followed by a 3x3 conv), so the
output is back at input resolution. Then a final conv with $T$ filters gives one cost map per
future timestep. The output is clipped to $[-1000, 1000]$.

## Trajectory sampler (inference)

The path is a **clothoid**: curvature is linear in arc length, $\kappa(\xi) = \pi\xi$ in canonical
form. This means steering at constant angular rate, which is consistent with a bicycle model
$\kappa = 2\tan\phi / L \approx 2\phi/L$.

$$
\mathbf{s}(\xi) = \mathbf{s}_0 + a\left[C\!\left(\tfrac{\xi}{a}\right)\mathbf{T}_0 + S\!\left(\tfrac{\xi}{a}\right)\mathbf{N}_0\right],
\qquad
S(x) = \int_0^x \sin\!\left(\tfrac{\pi u^2}{2}\right)du,\quad
C(x) = \int_0^x \cos\!\left(\tfrac{\pi u^2}{2}\right)du
$$

Velocity profile: constant acceleration, $\dot\xi(t) = \ddot\xi\, t + \dot\xi_0$.

Sampling recipe:
- Path type: straight with $p=0.5$, circle with $p=0.25$, clothoid with $p=0.25$. A clothoid
  handles neither of the first two well.
- Clothoid scale $a \sim U[6, 80]$ m, plus a random left/right mirror. The **current curvature**
  (from steering) picks the start position on the curve, so the plan is continuous with what the
  car is doing now.
- Acceleration $\ddot\xi \sim U[-5, 5]\ \text{m/s}^2$.
- Only one clothoid segment, which is judged enough for 3 s.

## Losses

$$
\mathcal{L} = \mathcal{L}_{\text{perception}} + \beta\, \mathcal{L}_{\text{planning}}
$$

**Perception.** Cross-entropy on the anchors (IoU > 0.4 assignment, unmatched GT goes to the
nearest anchor, hard negative mining) plus a weighted smooth-L1 regression, summed over all $t$:

$$
\mathcal{L}_{\text{perception}} = \sum \Big(\mathcal{L}_{cla} + \alpha \sum_{t=0}^{T}\mathcal{L}^t_{reg}\Big)
$$

**Planning (max-margin).** The GT ego trajectory is the positive. $N$ sampled trajectories are
negatives. With 0.8 probability a negative ignores the true initial state, which makes easy
negatives early in training. Writing $\hat c^t$ for the cost at the GT point and $c_i^t$ for the
cost at negative $i$:

$$
\mathcal{L}_{\text{planning}} = \sum_{\text{GT}} \max_{1\le i\le N} \sum_{t=1}^{T}\Big[\hat c^t - c_i^t + d_i^t + \gamma_i^t\Big]_+
$$

- $d_i^t = \lVert (x^t,y^t) - (x_i^t,y_i^t)\rVert_2$. Negatives far from the GT need a larger margin.
- $\gamma_i^t$ is a constant applied when negative $i$ breaks a traffic rule at $t$: a collision,
  running a red light, and so on. This is how safety gets in. Without it (ablation model 4)
  the 3 s collision rate goes from 0.78% to 1.40%.
- The ReLU sits **inside** the time sum, so no single timestep dominates. The max over negatives
  means only the hardest negative is optimized.

## Results that matter

The dataset is internal Uber data: 6.5k scenarios, 1.4M frames. Input is 10 Hz, output is
7 steps at 2 Hz (3 s).

| Method | L2 3s (m) | Collision 3s (%) | Lane viol. 3s (%) |
|---|---|---|---|
| Ego-motion MLP | 2.03 | 1.81 | 6.73 |
| Imitation learning | **1.92** | 1.72 | 5.38 |
| ACC | 2.80 | 2.39 | **0.64** |
| Manual cost + same sampler | 2.99 | 2.21 | 5.02 |
| **NMP** | 2.35 | **0.78** | 2.99 |

NMP is not best on L2. Imitation learning and the ego-only MLP beat it, because they optimize L2
directly. NMP's win is a collision rate 2-3x lower. The "manual cost" row uses NMP's sampler with
a hand-written cost (road 0, object boxes 255, everything else 100), so the gap between that row
and NMP is the value of the learned cost on its own.

## Mapping to this repo

### What the old `fsd/motion_planning/` did, and why it was replaced

Removed on 2026-11-04. Kept here as the motivation.

- `sampler.py`: 28 trajectories, 4 target speeds x 7 **constant** curvatures. Every candidate
  starts with zero heading rate whatever the current yaw rate is, so the curvature is
  discontinuous at $t=0$. There are no curvature transitions.
- `validator.py` checks every timestep against **one static** collision grid built at $t=0$.
  Moving actors are ignored, and so are the forecasts that `tracking.py` already computes
  (`TrackedObject.future_xy_ego`).
- `costs.py` is a hand-weighted sum in which $-4 \times$ progress dominates, so the planner
  mostly picks the fastest feasible arc. `lane_context` is always `None` in `runner.py`, so the map
  is never used.
- There is no evaluation against the logged ego trajectory: no L2, no collision rate. We cannot say
  whether a change makes it better.

### How NMP maps onto nuScenes

| NMP | nuScenes |
|---|---|
| 10 sweeps @ 10 Hz | LiDAR sweeps are 20 Hz. We take every other sweep, giving 10 sweeps over 1 s |
| Output 7 steps @ 2 Hz | Keyframes are 2 Hz. GT ego future = the next 6 keyframe ego poses (exact match) |
| Detection + forecast GT | `sample_annotation` linked across keyframes via `instance_token` |
| HD map channels | Map expansion layers: drivable_area, lane, road_divider, lane_divider, ped_crossing, walkway, stop_line, carpark. **There is no traffic light state** |
| $\gamma$ rule violations | Collision with GT future boxes, leaving the drivable area. Red light is not available |
| 1.4M frames | Full trainval has ~28k train keyframes. Our blobs 01-02 give ~4.4k (139 train scenes) plus 31 val scenes |

### Caveats

- **On nuScenes, open-loop L2 is dominated by ego status.** Zhai et al. 2023 (*Rethinking the
  Open-Loop Evaluation of E2E AD in nuScenes*) and Li et al., CVPR 2024 (*Is Ego Status All You
  Need?*) show that an MLP on ego velocity alone matches the perception-based planners on L2.
  Every comparison needs a constant-velocity baseline next to it, and **collision rate**, not
  L2, should be the headline metric. That is also NMP's own story.
- **There is a published reference to reproduce.** Khurana et al., ECCV 2022
  ([2210.01917](https://arxiv.org/abs/2210.01917), code: [peiyunh/ff](https://github.com/peiyunh/ff))
  re-implemented a simplified NMP on nuScenes-val with no map cost and no traffic-light cost.
  Box collision was 0.04 / 0.12 / 0.87 % and L2 was 0.53 / 1.25 / 2.67 m at 1 / 2 / 3 s. Their
  setup: 20 input sweeps (1 s at 20 Hz), the paper's 140.8 x 80 m ROI at 0.2 m, 1000 samples,
  7 outputs, Adam 5e-4, 15 epochs. Matching those numbers is how we know our implementation is
  correct.
- On an 8 GB GPU, a PIXOR backbone at $704 \times 400$ with $10 \times 27 + 8$ input channels
  needs mixed precision and batch size 2-4. Reading 10 raw sweeps per sample from disk will be the bottleneck, so the
  voxel and map tensors should be cached.
