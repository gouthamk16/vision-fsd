"""Build the sample cache, train and evaluate the neural motion planner.

The second documented CLI exception (after centerpoint_export.py): training is
not a visualization, so it does not belong in fsd.visualize.

    python -m fsd.nmp.train cache --split val
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fsd.data import NuScenesSceneLoader
from fsd.nmp.config import NmpConfig
from fsd.nmp.samples import build_scene_samples, save_sample, split_scenes
from fsd.nuscenes_map import NuScenesMapRenderer
from fsd.object_detection import NuScenesAnnotationLoader

DEFAULT_CACHE = Path.home() / ".cache" / "vision-fsd" / "nmp"


def build_cache(split: str, cache_dir: Path, dataroot: str | None, cfg: NmpConfig) -> int:
    loader = NuScenesSceneLoader(dataroot=dataroot)
    map_renderer = NuScenesMapRenderer(loader.dataroot)
    annotations = NuScenesAnnotationLoader(loader)
    out = cache_dir / split
    out.mkdir(parents=True, exist_ok=True)
    scenes = split_scenes(loader)[split]
    count = 0
    for i, scene in enumerate(scenes):
        for sample in build_scene_samples(loader, map_renderer, annotations, scene, cfg):
            save_sample(sample, out / f"{sample.token}.npz")
            count += 1
        print(f"[{i + 1}/{len(scenes)}] {scene}: {count} samples so far")
    return count


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    cache = sub.add_parser("cache", help="Build cached training samples for one split.")
    cache.add_argument("--split", choices=("train", "val"), required=True)
    cache.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    cache.add_argument("--dataroot", default=None)
    args = parser.parse_args(argv)
    if args.command == "cache":
        print(f"{build_cache(args.split, args.cache_dir, args.dataroot, NmpConfig())} samples written to {args.cache_dir / args.split}")


if __name__ == "__main__":
    main()
