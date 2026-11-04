from dataclasses import replace

import numpy as np

from fsd.nmp.config import NmpConfig
from fsd.nmp.samples import NmpSample, save_sample
from fsd.nmp.train import fit

SMALL = replace(NmpConfig(), x_range=(-8.0, 8.0), y_range=(-4.8, 4.8))


def _write_cache(root, rng):
    h, w = SMALL.grid.shape
    future = np.stack([2.0 * SMALL.times, np.zeros(6), np.zeros(6)], axis=1).astype(np.float32)  # 2 m/s straight
    for split in ("train", "val"):
        (root / split).mkdir(parents=True)
        for i in range(4):
            voxels = np.unique(rng.integers(0, SMALL.n_sweeps * SMALL.z_bins * h * w, 200)).astype(np.uint32)
            masks = np.zeros((8, h, w), np.uint8)
            masks[0] = 1  # all drivable: only the max-margin term shapes the cost
            sample = NmpSample(f"s{i}", "scene", voxels, masks, future, 2.0, 0.0,
                               np.zeros((0, 7, 5), np.float32), np.zeros((0, 7), bool), np.zeros(0, np.int8), np.zeros(0, np.int32))
            save_sample(sample, root / split / f"s{i}.npz")


def test_fit_overfits_four_samples(tmp_path):
    _write_cache(tmp_path / "cache", np.random.default_rng(0))
    history = fit(SMALL, tmp_path / "cache", tmp_path / "out", epochs=30, batch_size=4, lr=1e-3, workers=0, limit=4, device="cpu")
    assert history[-1]["plan"] <= 0.1 * history[0]["plan"]
    assert history[-1]["val"]["l2_3s"] < 1.0
    assert (tmp_path / "out" / "last.pt").exists() and (tmp_path / "out" / "best.pt").exists()
