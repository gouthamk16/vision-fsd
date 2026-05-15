from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from fsd.data import CAMERA_CHANNELS, NuScenesSceneLoader, SurroundFrame


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

        label = f"{channel}  t={camera.timestamp_us}"
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
    header = f"{frame.scene_name} | sample {frame.sample_index} | token {frame.sample_token}"
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


def save_contact_sheet(
    dataroot: str | Path | None = None,
    scene_index: int = 0,
    sample_index: int = 0,
    scene_name: str | None = None,
    output_path: str | Path = "outputs/nuscenes_contact_sheet.jpg",
    tile_width: int = 640,
) -> Path:
    loader = NuScenesSceneLoader(dataroot=dataroot)
    frame = loader.get_surround_frame(
        scene_index=scene_index,
        sample_index=sample_index,
        scene_name=scene_name,
    )
    sheet = render_contact_sheet(frame, tile_width=tile_width)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise OSError(f"Failed to write contact sheet: {output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a nuScenes six-camera contact sheet.")
    parser.add_argument("--dataroot", default=None, help="nuScenes root. Defaults to NUSCENES_ROOT or D:/nuscenes.")
    parser.add_argument("--scene-index", type=int, default=0, help="Scene index to render.")
    parser.add_argument("--scene-name", default=None, help="Scene name to render, e.g. scene-0001.")
    parser.add_argument("--sample-index", type=int, default=0, help="Sample index within the scene.")
    parser.add_argument("--tile-width", type=int, default=640, help="Width of each camera tile in pixels.")
    parser.add_argument("--output", default="outputs/nuscenes_contact_sheet.jpg", help="Output image path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = save_contact_sheet(
        dataroot=args.dataroot,
        scene_index=args.scene_index,
        sample_index=args.sample_index,
        scene_name=args.scene_name,
        output_path=args.output,
        tile_width=args.tile_width,
    )
    print(output)


if __name__ == "__main__":
    main()
