from __future__ import annotations

from dataclasses import replace
from typing import Callable

import cv2
import numpy as np

from fsd.data import CAMERA_CHANNELS, CameraFrame, SurroundFrame
from fsd.draw import label_band, with_header

# Draws on a tile and returns its label. The camera it receives has intrinsics
# scaled to the tile, so projections land directly in tile pixels.
TileAnnotator = Callable[[np.ndarray, CameraFrame], str]


def read_image(camera: CameraFrame) -> np.ndarray:
    image = cv2.imread(str(camera.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read camera image: {camera.image_path}")
    return image


def camera_sheet(
    frame: SurroundFrame,
    tile_width: int,
    title: str,
    subtitle: str = "",
    annotate: TileAnnotator | None = None,
) -> np.ndarray:
    """Tile the six cameras 3+3 under a header, annotating each tile."""
    tiles = []
    for channel in CAMERA_CHANNELS:
        camera = frame.cameras[channel]
        image = read_image(camera)
        scale = tile_width / image.shape[1]
        tile = cv2.resize(image, (tile_width, int(round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        tile_camera = replace(camera, intrinsic=np.diag([scale, scale, 1.0]) @ camera.intrinsic)
        label = annotate(tile, tile_camera) if annotate else channel
        label_band(tile, label)
        tiles.append(tile)
    return with_header(np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]), title, subtitle)


def render_contact_sheet(frame: SurroundFrame, tile_width: int = 640) -> np.ndarray:
    def annotate(_: np.ndarray, camera: CameraFrame) -> str:
        return f"{camera.channel}  {'key' if camera.is_key_frame else 'sweep'}  t={camera.timestamp_us}"

    x, y, z = frame.ego_pose.translation
    kind = "keyframe" if frame.is_key_frame else "sweep"
    return camera_sheet(
        frame,
        tile_width,
        f"{frame.scene_name} | {kind} {frame.sample_index} | token {frame.sample_token}",
        f"ego xyz=({x:.2f}, {y:.2f}, {z:.2f})",
        annotate,
    )
