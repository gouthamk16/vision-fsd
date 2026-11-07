from dataclasses import replace

import numpy as np
import pytest
import torch

from fsd.nmp import train
from fsd.nmp.config import NmpConfig
from fsd.nmp.samples import NmpSample, save_sample
from fsd.nmp.train import fit

SMALL = replace(NmpConfig(), x_range=(-8.0, 8.0), y_range=(-4.8, 4.8))


def _write_cache(root, rng, count=4):
    h, w = SMALL.grid.shape
    future = np.stack([2.0 * SMALL.times, np.zeros(6), np.zeros(6)], axis=1).astype(np.float32)  # 2 m/s straight
    for split in ("train", "val"):
        (root / split).mkdir(parents=True, exist_ok=True)
        for i in range(count):
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


def test_resume_after_a_crash_matches_an_uninterrupted_run(tmp_path, monkeypatch):
    _write_cache(tmp_path / "cache", np.random.default_rng(0))
    # A worker: persistent DataLoader workers draw RNG only when their iterator is first created.
    args = dict(epochs=2, batch_size=2, lr=1e-3, workers=1, limit=4, device="cpu")
    fit(SMALL, tmp_path / "cache", tmp_path / "full", **args)

    real, calls = train.train_epoch, []
    def crash_in_second_epoch(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(*a, **k)

    monkeypatch.setattr(train, "train_epoch", crash_in_second_epoch)
    with pytest.raises(KeyboardInterrupt):
        fit(SMALL, tmp_path / "cache", tmp_path / "resumed", **args)
    monkeypatch.undo()
    history = fit(SMALL, tmp_path / "cache", tmp_path / "resumed", resume=True, **args)

    assert len(history) == 1
    full, resumed = (torch.load(tmp_path / d / "last.pt", weights_only=True)["model"] for d in ("full", "resumed"))
    assert all(torch.equal(full[k], resumed[k]) for k in full)


def test_resume_refuses_a_missing_checkpoint_or_a_changed_run(tmp_path):
    _write_cache(tmp_path / "cache", np.random.default_rng(0))
    args = dict(batch_size=2, lr=1e-3, workers=0, limit=4, device="cpu")
    with pytest.raises(FileNotFoundError, match="last.pt"):
        fit(SMALL, tmp_path / "cache", tmp_path / "out", epochs=1, resume=True, **args)
    fit(SMALL, tmp_path / "cache", tmp_path / "out", epochs=1, **args)
    with pytest.raises(ValueError, match="epochs"):
        fit(SMALL, tmp_path / "cache", tmp_path / "out", epochs=3, resume=True, **args)


def test_resume_refuses_a_rebuilt_train_set(tmp_path):
    _write_cache(tmp_path / "cache", np.random.default_rng(0))
    args = dict(epochs=2, batch_size=2, lr=1e-3, workers=0, device="cpu")
    fit(SMALL, tmp_path / "cache", tmp_path / "out", **args)
    _write_cache(tmp_path / "cache", np.random.default_rng(1), count=6)  # e.g. another blob extracted, cache rebuilt
    with pytest.raises(ValueError, match="train_samples"):
        fit(SMALL, tmp_path / "cache", tmp_path / "out", resume=True, **args)
