"""Torch dataset over the cached samples: densify inputs, rasterise future obstacles."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from fsd.bev import SHIFT, fixed_point
from fsd.nmp.config import NmpConfig
from fsd.nmp.samples import NmpSample, load_sample
from fsd.object_detection import Box3D

# Obstacles grow by ~the ego half-width, so a trajectory point (the ego body
# centre) inside them means the bodies overlap.
_DILATE_CELLS = 5


def footprint(box_xywly: np.ndarray) -> np.ndarray:
    """(4, 2) corners of one (x, y, w, l, yaw) box row, in Box3D corner order."""
    x, y, w, l, yaw = (float(v) for v in box_xywly)
    return Box3D("", np.array([x, y, 0.0]), np.array([w, l, 1.0]), yaw).footprint


def obstacle_masks(sample: NmpSample, cfg: NmpConfig) -> np.ndarray:
    """(T, H, W) bool: every box valid at future step k, dilated by the ego half-width."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * _DILATE_CELLS + 1,) * 2)
    masks = np.zeros((cfg.horizon, *cfg.grid.shape), np.uint8)
    for step in range(1, cfg.horizon + 1):
        polygons = [fixed_point(cfg.grid, footprint(sample.boxes[row, step])) for row in np.flatnonzero(sample.box_valid[:, step])]
        if polygons:
            cv2.fillPoly(masks[step - 1], polygons, 1, cv2.LINE_8, SHIFT)
            masks[step - 1] = cv2.dilate(masks[step - 1], kernel)
    return masks.astype(bool)


class NmpDataset(Dataset):
    def __init__(self, cache_dir: Path, split: str, cfg: NmpConfig):
        self.paths = sorted((Path(cache_dir) / split).glob("*.npz"))
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = load_sample(self.paths[index])
        h, w = self.cfg.grid.shape
        voxels = np.zeros((self.cfg.n_sweeps * self.cfg.z_bins) * h * w, np.uint8)
        voxels[sample.voxels] = 1
        dense = np.concatenate([voxels.reshape(-1, h, w), sample.map_masks])
        return {
            "input": torch.from_numpy(dense),
            "ego_future": torch.from_numpy(sample.ego_future),
            "v0": torch.tensor(sample.v0, dtype=torch.float32),
            "k0": torch.tensor(sample.k0, dtype=torch.float32),
            "boxes": torch.from_numpy(sample.boxes),
            "box_valid": torch.from_numpy(sample.box_valid),
            "box_class": torch.from_numpy(sample.box_class.astype(np.int64)),
            "obstacles": torch.from_numpy(obstacle_masks(sample, self.cfg)),
            "drivable": torch.from_numpy(sample.map_masks[0].astype(bool)),
        }


def collate(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack a batch, padding the per-sample box tables to the largest K (padding is invalid)."""
    k = max(item["boxes"].shape[0] for item in items)
    batch = {key: torch.stack([item[key] for item in items]) for key in items[0] if not key.startswith("box")}
    for key in ("boxes", "box_valid", "box_class"):
        padded = []
        for item in items:
            pad = k - item[key].shape[0]
            padded.append(torch.cat([item[key], item[key].new_zeros((pad, *item[key].shape[1:]))]))
        batch[key] = torch.stack(padded)
    return batch
