## Vision for FSD

Fully vision-based path tracking and environment mapping for autonomous driving.

Part of a larger FSD research stack: [Monocular SLAM](https://github.com/gouthamk16/Slam) · [Environment Reasoning](https://github.com/gouthamk16/drive-vlm) · [DL Based Actuators](https://github.com/gouthamk16/xdrive)

---

## What this does

Each video frame is passed through three subsystems:

| Subsystem | Model | Output |
|---|---|---|
| Object detection + tracking | YOLOv2.6 + ByteTrack | 2D bounding boxes with locked class labels and distance labels |
| Monocular depth estimation | DepthAnything V2 (metric outdoor) | Per-pixel depth map in metres |
| Visual odometry | SIFT + FLANN + Essential matrix | Camera pose and trajectory |

Everything is composed in a bird's-eye-view (BEV) minimap showing detected vehicles at real-world scale up to ~83 m.

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

| File | Size | Purpose |
|---|---|---|
| `yolo26n.pt` | ~6 MB | Vehicle detection (included in repo) |
| DepthAnything V2 (metric outdoor large) | ~335 MB | Depth estimation (auto-downloaded from HuggingFace on first run) |

---

## Todo

- Extended Kalman filters for trajectory mapping and path tracking
- LANES
