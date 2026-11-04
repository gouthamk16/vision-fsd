"""Build the sample cache, train and evaluate the neural motion planner.

The second documented CLI exception (after centerpoint_export.py): training is
not a visualization, so it does not belong in fsd.visualize.

    python -m fsd.nmp.train cache --split val
    python -m fsd.nmp.train fit
    python -m fsd.nmp.train eval --weights checkpoints/nmp/best.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from fsd.data import NuScenesSceneLoader
from fsd.nmp.config import NmpConfig
from fsd.nmp.dataset import make_loader
from fsd.nmp.evaluate import evaluate, report
from fsd.nmp.losses import detection_loss, detection_targets, planning_loss, violations
from fsd.nmp.model import NeuralMotionPlanner
from fsd.nmp.samples import build_scene_samples, save_sample, split_scenes
from fsd.nmp.trajectories import sample_trajectories
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


def train_step(model: NeuralMotionPlanner, batch: dict, cfg: NmpConfig, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """(detection loss, planning loss) for one batch already on the model's device."""
    device = batch["input"].device
    with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
        out = model(batch["input"])
    det = detection_loss(out, detection_targets(batch["boxes"], batch["box_valid"], batch["box_class"], cfg))
    negatives = sample_trajectories(batch["v0"], batch["k0"], cfg, cfg.n_trajectories, 1.0 - cfg.negative_free_init_p, generator)
    gamma = violations(negatives, batch["obstacles"], batch["drivable"], cfg)
    return det, planning_loss(out.cost.float(), batch["ego_future"], negatives, gamma, cfg.grid)


def train_epoch(model, loader, optimizer, scheduler, cfg: NmpConfig, generator, log_every: int = 50) -> dict[str, float]:
    model.train()
    device = next(model.parameters()).device
    totals = {"det": 0.0, "plan": 0.0, "total": 0.0}
    for step, batch in enumerate(loader, 1):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        det, plan = train_step(model, batch, cfg, generator)
        loss = det + cfg.beta * plan
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        for key, value in (("det", det), ("plan", plan), ("total", loss)):
            totals[key] += float(value)
        if step % log_every == 0:
            print(f"  step {step}/{len(loader)} det {float(det):.3f} plan {float(plan):.1f} total {float(loss):.3f}")
    return {key: value / len(loader) for key, value in totals.items()}


def fit(cfg: NmpConfig, cache_dir: Path, out_dir: Path, epochs: int = 20, batch_size: int = 4, lr: float = 2e-4,
        workers: int = 4, limit: int | None = None, device: str | None = None, seed: int = 0) -> list[dict]:
    """Train; save last.pt every epoch and best.pt by val collision at 3 s (then L2 at 3 s)."""
    torch.manual_seed(seed)  # model init and shuffling: runs are reproducible
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train = make_loader(cache_dir, "train", cfg, batch_size, workers, limit, shuffle=True)
    val = make_loader(cache_dir, "val", cfg, batch_size, workers, limit, shuffle=False)
    model = NeuralMotionPlanner(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train))
    generator = torch.Generator(device=device).manual_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    history, best = [], None
    for epoch in range(epochs):
        start = time.time()
        stats = train_epoch(model, train, optimizer, scheduler, cfg, generator)
        stats["val"] = evaluate(model, val, cfg)
        history.append(stats)
        key = (stats["val"]["collision_3s"], stats["val"]["l2_3s"])
        torch.save({"model": model.state_dict(), "epoch": epoch, "val": stats["val"]}, out_dir / "last.pt")
        if best is None or key < best:
            best = key
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": stats["val"]}, out_dir / "best.pt")
        print(f"epoch {epoch + 1}/{epochs} ({time.time() - start:.0f}s) det {stats['det']:.3f} plan {stats['plan']:.1f} "
              f"| val L2@3s {stats['val']['l2_3s']:.2f} m coll@3s {stats['val']['collision_3s']:.2f}%")
    return history


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    cache = sub.add_parser("cache", help="Build cached training samples for one split.")
    cache.add_argument("--split", choices=("train", "val"), required=True)
    cache.add_argument("--dataroot", default=None)
    train = sub.add_parser("fit", help="Train the planner.")
    train.add_argument("--out", type=Path, default=Path("checkpoints/nmp"))
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--lr", type=float, default=2e-4)
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--limit", type=int, default=None, help="Use only the first N samples of each split.")
    ev = sub.add_parser("eval", help="Report planning and detection metrics against baselines.")
    ev.add_argument("--weights", type=Path, required=True)
    ev.add_argument("--split", choices=("train", "val"), default="val")
    for p in (cache, train, ev):
        p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args(argv)
    cfg = NmpConfig()
    if args.command == "cache":
        print(f"{build_cache(args.split, args.cache_dir, args.dataroot, cfg)} samples written to {args.cache_dir / args.split}")
    elif args.command == "fit":
        fit(cfg, args.cache_dir, args.out, args.epochs, args.batch_size, args.lr, args.workers, args.limit)
    else:
        print(report(args.weights, args.cache_dir, args.split, cfg))


if __name__ == "__main__":
    main()
