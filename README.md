## Vision for FSD 

Vision-based path tracking and environment mapping for autonomous vehicles. Extracting important features from monocular frames for advanced driver assistance systems. Currently also working on interpreting features from surround vision and lidar inputs.

**Implemented using geometry based feature extractors and open-sourced models. Not practically useful, more of a research oriented project.**

Part of a larger FSD research stack: [Monocular SLAM](https://github.com/gouthamk16/Slam) · [Environment Reasoning](https://github.com/gouthamk16/drive-vlm) · [DL Based Actuators](https://github.com/gouthamk16/xdrive)

The core idea of this project is to build an end-to-end perception model that leverages these features to predict driving trajectories and actuator-level control signals (steering angle, throttle input, and braking intensity etc) directly from sensor observations.

A major challenge in developing such systems is the massive amounts of annotated driving data required. Modern autonomous driving systems rely heavily on large-scale real-world data collection pipelines. For example, Tesla continuously gathers driving data from customer vehicles in real time, enabling iterative large-scale training and refinement of its Autopilot models (they call it "Fleet-Learning"). This continuous learning approach provides access to highly diverse driving scenarios and edge cases which significantly improves model generalization.

---

## What the monocular vision extractor does

Each video frame is passed through four subsystems running in sequence:

| Subsystem | Model | Output |
|---|---|---|
| Object detection + tracking | YOLOv2.6 + ByteTrack | 2D bounding boxes with locked class labels and per-object distance |
| Monocular depth estimation | DepthAnything V2 (metric outdoor) | Per-pixel depth map in metres |
| Drivable area segmentation | YOLOPv2 | Binary pixel mask separating drivable road from non-drivable regions |
| Visual odometry | SIFT + FLANN + Essential matrix | Camera pose and trajectory |

Depth and segmentation outputs are fused in the final render: the drivable area is overlaid as a green tint on the frame, detected vehicles get per-object distance labels from the depth map, and everything is summarised in a bird's-eye-view (BEV) minimap covering ~83 m forward range.

## Results

**Detection + drivable area segmentation + BEV minimap**

![Detection and segmentation result](results/result1.png)

**Depth estimation (Note: This is a different frame from the one above)**

![Depth estimation result](results/result-depth.png)

**Another interesting result, you can see how it understands double yellow lines while segmenting drivable area and only considers the current lane**

![Drivable Area NYC](results/city.png)

---

## 360 Vision - nuScenes 

Beyond single-camera dashcam footage, we also work with the [nuScenes](https://www.nuscenes.org/) dataset, which has six surround-view cameras + LiDAR on a real autonomous vehicle. This lets us look in all directions at once and work with proper 3D sensor data instead of inferring depth from a single image. Implemented in the [fsd](fsd/) folder.

nuScenes is organised as ~850 independent 20-second driving clips (called scenes), each with ~40 keyframes at 2 Hz. They're not continuous - scene 5 has nothing to do with scene 6 temporally or geographically, but each scene on its own is a clean, calibrated, fully-labelled snippet of real urban driving.

### What we can visualise

| View | What it shows |
|---|---|
| `cameras` | All six cameras tiled into a contact sheet |
| `lidar` | LiDAR point cloud projected onto each camera image, coloured by depth |
| `bev` | Bird's-eye-view of the LiDAR sweep from above |
| `lss_bev` | Camera-only BEV vehicle prediction from Lift-Splat-Shoot, drawn on top of the nuScenes HD map |
| `lss_lidar_bev` | LSS prediction blended onto the LiDAR BEV for a direct sanity check |
| `all` | All views rendered simultaneously |

### Camera-only BEV with Lift-Splat-Shoot

The six surround cameras alone are enough to estimate a top-down vehicle occupancy map. We integrate NVIDIA's [Lift-Splat-Shoot](https://github.com/nv-tlabs/lift-splat-shoot) (ECCV 2020): each camera image is lifted into a per-pixel depth distribution, splat into a shared ego-frame voxel grid, and decoded into a BEV vehicle segmentation. The pretrained checkpoint runs without finetuning, and we render the output on top of the nuScenes HD map expansion pack (includes lanes, road outlines, lane lines etc). Implementation in [fsd/lss.py](fsd/lss.py) and [fsd/nuscenes_map.py](fsd/nuscenes_map.py).

### Results

**Six surround cameras tiled into a contact sheet**

![nuScenes cameras](results/nuscenes_cameras.jpg)

**LiDAR point cloud projected onto the camera images, coloured by depth**

![nuScenes lidar projection](results/nuscenes_lidar.jpg)

**Bird's-eye-view of the LiDAR sweep**

![nuScenes BEV](results/nuscenes_bev.jpg)

**Lift-Splat-Shoot vehicle BEV prediction overlaid on the nuScenes HD map (singapore-onenorth)**

![nuScenes LSS BEV](results/nuscenes_lss_bev.jpg)

### Run

```bash
# Single scene, default 20 keyframes
python -m fsd.visualize --view cameras --save

# All frames in a scene
python -m fsd.visualize --view lidar --frames all --save

# Stitch scenes 0 through 9 into one video
python -m fsd.visualize --scenes 0-9 --frames all --view cameras --save

# Stream all three views at once
python -m fsd.visualize --view all --frames all --stream

# Smoother video using raw camera sweeps (~12 Hz instead of 2 Hz keyframes)
python -m fsd.visualize --view cameras --sequence sweeps --frames all --save
```

nuScenes data should live at `D:/nuscenes` or set `NUSCENES_ROOT` to point elsewhere.

---

## Requirements

- Python 3.12
- CUDA-capable GPU (tested on RTX 4060, CUDA 12.8)

---

## Setup

### 1. Clone and create environment

```bash
git clone <this-repo>
cd vision-fsd
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 2. Install PyTorch with CUDA

Install the CUDA 12.8 build from pytorch.org. The requirements.txt pins the `+cu128` variants, so install those first before the rest of the deps:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your video

Place a dashcam video at `data/` and pass the path at runtime.

---

## Run

```bash
# Stream processed output to screen
python main.py data/your_video.mp4 --stream

# Save annotated video to disk
python main.py data/your_video.mp4 --save

# Limit to first N frames (useful for quick tests)
python main.py data/your_video.mp4 --stream --frames 300

# Full options
python main.py --help
```

---

## Models

| File | Size | Source |
|---|---|---|
| `yolo26n.pt` | ~6 MB | Included in repo |
| `yolopv2.pt` | ~70 MB | Auto-downloaded from GitHub releases on first run |
| DepthAnything V2 (metric outdoor large) | ~335 MB | Auto-downloaded from HuggingFace on first run |

---

## Todo

- Extended Kalman filters for trajectory mapping and path tracking
- Lane detection
