from __future__ import annotations

import cv2
import numpy as np

from fsd.data import CAMERA_CHANNELS, SurroundFrame


def render_contact_sheet(frame: SurroundFrame, tile_width: int = 640) -> np.ndarray:
    tiles = []
    for channel in CAMERA_CHANNELS:
        camera = frame.cameras[channel]
        image = cv2.imread(str(camera.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read camera image: {camera.image_path}")

        height, width = image.shape[:2]
        scale = tile_width / width
        tile_size = (tile_width, int(round(height * scale)))
        tile = cv2.resize(image, tile_size, interpolation=cv2.INTER_AREA)

        frame_type = "key" if camera.is_key_frame else "sweep"
        label = f"{channel}  {frame_type}  t={camera.timestamp_us}"
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(
            tile,
            label,
            (10, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    top = np.hstack(tiles[:3])
    bottom = np.hstack(tiles[3:])

    header_h = 76
    sheet = np.zeros((header_h + top.shape[0] + bottom.shape[0], top.shape[1], 3), dtype=np.uint8)
    sheet[:] = (22, 22, 22)

    ego = frame.ego_pose
    frame_type = "keyframe" if frame.is_key_frame else "sweep"
    header = f"{frame.scene_name} | {frame_type} {frame.sample_index} | token {frame.sample_token}"
    pose = (
        f"ego xyz=({ego['translation'][0]:.2f}, {ego['translation'][1]:.2f}, "
        f"{ego['translation'][2]:.2f})"
    )
    cv2.putText(sheet, header, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(sheet, pose, (12, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (190, 220, 255), 1, cv2.LINE_AA)

    y = header_h
    sheet[y:y + top.shape[0], :] = top
    y += top.shape[0]
    sheet[y:y + bottom.shape[0], :] = bottom
    return sheet
