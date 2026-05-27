"""
3D visualization for nuScenes LiDAR point cloud data.

Usage:
    python lidar_3d_viewer.py                          # scene 0, sample 0
    python lidar_3d_viewer.py --scene 3 --sample 10    # specific scene/sample
    python lidar_3d_viewer.py --animate                # animate all keyframes in scene
    python lidar_3d_viewer.py --animate --scene 2 --fps 5
    python lidar_3d_viewer.py --file path/to/points.pcd.bin
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np



def load_pcd_bin(path: str | Path) -> np.ndarray:
    pts = np.fromfile(str(path), dtype=np.float32)
    if pts.size % 5 != 0:
        raise ValueError(f"Unexpected point format: {pts.size} floats (expected multiple of 5)")
    return pts.reshape(-1, 5)


def get_loader(dataroot):
    sys.path.insert(0, str(Path(__file__).parent))
    from fsd.data import NuScenesSceneLoader
    return NuScenesSceneLoader(dataroot=dataroot)



def colour_by_z(points: np.ndarray, z_min=-3.0, z_max=5.0) -> np.ndarray:
    import matplotlib.cm as cm
    t = np.clip((points[:, 2] - z_min) / (z_max - z_min), 0, 1)
    return cm.turbo(t)[:, :3]


def colour_by_distance(points: np.ndarray, max_dist=80.0) -> np.ndarray:
    import matplotlib.cm as cm
    d = np.linalg.norm(points[:, :2], axis=1)
    t = 1.0 - np.clip(d / max_dist, 0, 1)
    return cm.turbo(t)[:, :3]


def colourise(points, mode):
    return colour_by_z(points) if mode == "z" else colour_by_distance(points)



def _make_vis(title="LiDAR 3D Viewer"):
    import open3d as o3d
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=720)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.05])
    opt.point_size = 3.0
    return vis


def view_open3d(points: np.ndarray, colours: np.ndarray) -> None:
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colours.astype(np.float64))
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)

    print(f"{len(points):,} points — drag=rotate  scroll=zoom  shift+drag=pan  Q=quit")
    vis = _make_vis()
    vis.add_geometry(pcd)
    vis.add_geometry(frame)
    vis.run()
    vis.destroy_window()


def animate_open3d(scene_index: int, colour_mode: str, dataroot, fps: float) -> None:
    import open3d as o3d

    loader = get_loader(dataroot)
    frame_delay = 1.0 / fps

    print(f"Animating scene {scene_index} at {fps} fps — Q=quit")

    vis = _make_vis(f"LiDAR Animate — scene {scene_index}")
    frame_geom = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
    vis.add_geometry(frame_geom)

    pcd = o3d.geometry.PointCloud()
    first = True

    for surround, lidar in loader.iter_scene_frames(scene_index=scene_index, include_lidar=True):
        if not vis.poll_events():
            break

        raw = load_pcd_bin(lidar.pointcloud_path)
        points = raw[:, :3].astype(np.float64)
        colours = colourise(points, colour_mode).astype(np.float64)

        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colours)

        if first:
            vis.add_geometry(pcd)
            first = False
        else:
            vis.update_geometry(pcd)

        print(f"  sample {surround.sample_index:02d} — {len(points):,} pts", end="\r")

        vis.update_renderer()
        time.sleep(frame_delay)

    print()
    vis.destroy_window()


def view_matplotlib(points: np.ndarray, colours: np.ndarray, max_points=15_000) -> None:
    import matplotlib.pyplot as plt
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points, colours = points[idx], colours[idx]
        print(f"Downsampled to {max_points:,} points (install open3d for full cloud)")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colours, s=0.5, linewidths=0)
    ax.set_xlabel("X (forward, m)")
    ax.set_ylabel("Y (left, m)")
    ax.set_zlabel("Z (up, m)")
    ax.set_box_aspect([1, 1, 0.15])
    plt.tight_layout()
    plt.show()


def parse_args():
    p = argparse.ArgumentParser(description="View nuScenes LiDAR points in 3D.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--file", help="Path to a .pcd.bin file directly.")
    src.add_argument("--scene", type=int, default=0, help="Scene index (default 0).")
    p.add_argument("--sample", type=int, default=0, help="Sample index within the scene (default 0).")
    p.add_argument("--animate", action="store_true", help="Animate all keyframes in the scene.")
    p.add_argument("--fps", type=float, default=2.0, help="Playback speed for --animate (default 2.0).")
    p.add_argument("--dataroot", default=None, help="nuScenes root. Defaults to NUSCENES_ROOT or D:/nuscenes.")
    p.add_argument("--colour", choices=("z", "distance"), default="z", help="Colour by height (z) or horizontal distance.")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        import open3d
        has_open3d = True
    except ImportError:
        has_open3d = False

    if args.animate:
        if not has_open3d:
            print("--animate requires open3d: pip install open3d")
            return
        animate_open3d(args.scene, args.colour, args.dataroot, args.fps)
        return

    if args.file:
        raw = load_pcd_bin(args.file)
    else:
        loader = get_loader(args.dataroot)
        lidar = loader.get_lidar_frame(scene_index=args.scene, sample_index=args.sample)
        print(f"Scene {args.scene}, sample {args.sample}: {lidar.pointcloud_path}")
        raw = load_pcd_bin(lidar.pointcloud_path)

    points = raw[:, :3]
    print(f"Points: {len(points):,}  |  X [{points[:,0].min():.1f}, {points[:,0].max():.1f}]"
          f"  Y [{points[:,1].min():.1f}, {points[:,1].max():.1f}]"
          f"  Z [{points[:,2].min():.1f}, {points[:,2].max():.1f}]")

    colours = colourise(points, args.colour)

    if has_open3d:
        view_open3d(points, colours)
    else:
        print("open3d not found — using matplotlib (pip install open3d for better experience)")
        view_matplotlib(points, colours)


if __name__ == "__main__":
    main()
