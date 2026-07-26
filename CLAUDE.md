# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vision-based Full Self-Driving pipeline that processes dashcam video through three parallel subsystems — object detection/tracking, monocular depth estimation, and visual odometry — then fuses their outputs into an annotated video with a bird's-eye-view (BEV) minimap.

## Environment Setup

Requires Python 3.12, CUDA 12.8, and an activated `.venv`:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

DepthAnything V2 weights are auto-downloaded from HuggingFace on first run. YOLO weights (`yolo26n.pt`) must be present at the repo root.

## Running

```bash
# Stream to screen (press 'q' to exit)
python main.py data/your_video.mp4 --stream

# Save annotated output to outputs/
python main.py data/your_video.mp4 --save

# Quick test on first N frames
python main.py data/your_video.mp4 --stream --frames 300
```

Outputs land in `outputs/` (annotated video + depth video) and `logs/`. Log verbosity is controlled by the `LOGGING_LEVEL` env var (default: INFO).

## Architecture

**Entry point**: `main.py` → `fsd/fsd.py` → `fsd/process_frame.py`

`fsd.py` handles video I/O and frame resampling (>45 FPS input is resampled to 20 FPS target). Each frame is passed to `FrameProcessor.process()`.

**Three subsystems run per frame inside `FrameProcessor`:**

| Module | Class | Responsibility |
|---|---|---|
| `vision.py` | `VisualOdometry` | SIFT → FLANN → Essential matrix pose recovery; outputs camera trajectory |
| `detect.py` | `VehicleTracker` | YOLOv2.6 + ByteTrack; outputs 2D bboxes and track IDs |
| `depth.py` | `MonocularDepthEstimator` | DepthAnything V2 metric outdoor; outputs per-pixel depth in meters |

**BEV rendering** (`detect.py`): Projects 2D bboxes + per-object depth into 3D world coordinates. Renders a 320×440 px minimap covering ~83 m forward and ±32 m lateral. Class-specific depth extents are hardcoded priors (Car=1.8 m, Truck=12 m, Bus=2.2 m, Bicycle/Motorcycle=0.4–4.5 m).

**Tracked vehicle classes** (COCO IDs): Bicycle(1), Car(2), Motorcycle(3), Bus(5), Truck(7).

`enhance.py` contains low-light image decomposition — partially implemented, not yet integrated into the main pipeline.

## No Tests or Linter

There is no test suite or linting config. Validate changes by running the pipeline on a short clip with `--frames 300`.
