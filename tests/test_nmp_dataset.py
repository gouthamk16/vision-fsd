import numpy as np
import torch

from fsd.nmp.config import NmpConfig
from fsd.nmp.dataset import NmpDataset, collate
from fsd.nmp.samples import NmpSample, save_sample

CFG = NmpConfig()
H, W = CFG.grid.shape


def _sample(token: str, k: int) -> NmpSample:
    masks = np.zeros((8, H, W), np.uint8)
    masks[0, 300:400, 150:250] = 1
    boxes = np.zeros((k, 7, 5), np.float32)
    boxes[:, :, :] = [20.0, 0.0, 2.0, 4.0, 0.0]
    valid = np.zeros((k, 7), bool)
    valid[:, :2] = True  # present now and at step 1 only
    return NmpSample(token, "scene", np.array([0, 5, (27 * H + 3) * W + 7], np.uint32), masks,
                     np.zeros((6, 3), np.float32), 3.0, 0.0, boxes, valid, np.zeros(k, np.int8), np.ones(k, np.int32))


def _dataset(tmp_path, samples):
    (tmp_path / "val").mkdir()
    for s in samples:
        save_sample(s, tmp_path / "val" / f"{s.token}.npz")
    return NmpDataset(tmp_path, "val", CFG)


def test_densify_matches_voxel_indices(tmp_path):
    item = _dataset(tmp_path, [_sample("a", 1)])[0]
    assert item["input"].shape == (278, H, W) and item["input"].dtype == torch.uint8
    assert int(item["input"].sum()) == 3 + 100 * 100
    assert item["input"][27, 3, 7] == 1 and item["input"][270, 350, 200] == 1
    assert bool(item["drivable"][350, 200]) and not bool(item["drivable"][0, 0])


def test_obstacles_only_where_box_valid(tmp_path):
    item = _dataset(tmp_path, [_sample("a", 1)])[0]
    rows, cols, _ = CFG.grid.cells(np.array([[20.0, 0.0]]))
    assert item["obstacles"].shape == (6, H, W)
    assert bool(item["obstacles"][0, rows[0], cols[0]]) and not item["obstacles"][1].any()
    # Dilated by ~1 m: a cell 1.9 m beside the 2 m-wide box is blocked, 3 m is not.
    r, c, _ = CFG.grid.cells(np.array([[20.0, 1.9], [20.0, 3.0]]))
    assert bool(item["obstacles"][0, r[0], c[0]]) and not bool(item["obstacles"][0, r[1], c[1]])


def test_collate_pads_boxes(tmp_path):
    ds = _dataset(tmp_path, [_sample("a", 2), _sample("b", 0)])
    batch = collate([ds[0], ds[1]])
    assert batch["boxes"].shape == (2, 2, 7, 5) and batch["input"].shape == (2, 278, H, W)
    assert not batch["box_valid"][1].any() and batch["box_valid"][0, :, :2].all()
