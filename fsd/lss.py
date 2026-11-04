"""Lift-Splat-Shoot model + inference wrapper for our nuScenes pipeline.

Model code (CamEncode, BevEncode, Up, LiftSplatShoot) and the QuickCumsum
trick are direct ports of the original NVIDIA Lift-Splat-Shoot repo
(https://github.com/nv-tlabs/lift-splat-shoot, src/models.py and src/tools.py).

The inference wrapper takes our `SurroundFrame` instead of going through the
nuscenes-devkit dataloader. It runs the pretrained BEV vehicle segmentation
checkpoint released by the authors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from efficientnet_pytorch import EfficientNet
from torch import nn
from torchvision.models.resnet import resnet18

from fsd.bev import BevGrid, canvas, draw_ego, draw_metric_grid, resample
from fsd.data import CAMERA_CHANNELS, SurroundFrame
from fsd.draw import blend, with_header
from fsd.nuscenes_map import NuScenesMapRenderer

# The released checkpoint's configuration (src/explore.py defaults upstream).
GRID_CONF = {
    "xbound": [-50.0, 50.0, 0.5],
    "ybound": [-50.0, 50.0, 0.5],
    "zbound": [-10.0, 10.0, 20.0],
    "dbound": [4.0, 45.0, 1.0],
}
DATA_AUG_CONF = {"final_dim": (128, 352), "H": 900, "W": 1600, "bot_pct_lim": (0.0, 0.22)}
LSS_GRID = BevGrid((-50.0, 50.0), (-50.0, 50.0), 0.5)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = ranks[1:] != ranks[:-1]
        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))
        ctx.save_for_backward(kept)
        ctx.mark_non_differentiable(geom_feats)
        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        (kept,) = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1
        return gradx[back], None, None


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, D, C):
        super().__init__()
        self.D = D
        self.C = C
        # Architecture only: the full LSS checkpoint is loaded right after.
        self.trunk = EfficientNet.from_name("efficientnet-b0")
        self.up1 = Up(320 + 112, 512)
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_feat(self, x):
        x = self.get_eff_depth(x)
        x = self.depthnet(x)
        depth = x[:, : self.D].softmax(dim=1)
        new_x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)
        return depth, new_x

    def get_eff_depth(self, x):
        endpoints = {}
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints[f"reduction_{len(endpoints) + 1}"] = prev_x
            prev_x = x
        endpoints[f"reduction_{len(endpoints) + 1}"] = x
        return self.up1(endpoints["reduction_5"], endpoints["reduction_4"])

    def forward(self, x):
        _, x = self.get_depth_feat(x)
        return x


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super().__init__()
        trunk = resnet18(weights=None, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu
        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3
        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)
        x = self.up1(x, x1)
        return self.up2(x)


class LiftSplatShoot(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC):
        super().__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf

        dx, bx, nx = gen_dx_bx(grid_conf["xbound"], grid_conf["ybound"], grid_conf["zbound"])
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC)
        self.bevencode = BevEncode(inC=self.camC, outC=outC)

    def create_frustum(self):
        ogfH, ogfW = self.data_aug_conf["final_dim"]
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_conf["dbound"], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        B, N, _ = trans.shape
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        return points

    def get_cam_feats(self, x):
        B, N, C, imH, imW = x.shape
        x = x.view(B * N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, N, self.camC, self.D, imH // self.downsample, imW // self.downsample)
        return x.permute(0, 1, 3, 4, 5, 2)

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        x = x.reshape(Nprime, C)
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]
        ranks = (
            geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)
            + geom_feats[:, 1] * (self.nx[2] * B)
            + geom_feats[:, 2] * B
            + geom_feats[:, 3]
        )
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]
        x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x
        return torch.cat(final.unbind(dim=2), 1)

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x = self.get_cam_feats(x)
        return self.voxel_pooling(geom, x)

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        return self.bevencode(x)


def _eval_image_transform(image: Image.Image, data_aug_conf: dict) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
    """Replicates the eval-time (resize, crop) used in src/data.py."""
    H, W = data_aug_conf["H"], data_aug_conf["W"]
    fH, fW = data_aug_conf["final_dim"]
    resize = max(fH / H, fW / W)
    resize_dims = (int(W * resize), int(H * resize))
    newW, newH = resize_dims
    crop_h = int((1 - float(np.mean(data_aug_conf["bot_pct_lim"]))) * newH) - fH
    crop_w = int(max(0, newW - fW) / 2)
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

    image = image.resize(resize_dims)
    image = image.crop(crop)

    post_rot = torch.eye(2) * resize
    post_tran = torch.zeros(2) - torch.Tensor(crop[:2])

    post_rot_3 = torch.eye(3)
    post_tran_3 = torch.zeros(3)
    post_rot_3[:2, :2] = post_rot
    post_tran_3[:2] = post_tran
    return image, post_rot_3, post_tran_3


def _normalize_image(image: Image.Image) -> torch.Tensor:
    tensor = TF.to_tensor(image)
    return TF.normalize(tensor, mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD))


class LSSInference:
    """Run the pretrained LSS BEV vehicle-segmentation model on `SurroundFrame`s."""

    def __init__(self, weights_path: str | Path, device: str | torch.device | None = None):
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"LSS weights not found: {weights_path}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = LiftSplatShoot(GRID_CONF, DATA_AUG_CONF, outC=1)
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        self.model.to(self.device).eval()

    def _inputs(self, frame: SurroundFrame) -> list[torch.Tensor]:
        imgs, rots, trans, intrins, post_rots, post_trans = [], [], [], [], [], []
        for channel in CAMERA_CHANNELS:
            camera = frame.cameras[channel]
            image, post_rot, post_tran = _eval_image_transform(Image.open(camera.image_path).convert("RGB"), DATA_AUG_CONF)
            imgs.append(_normalize_image(image))
            rots.append(torch.tensor(camera.calibration.rotation, dtype=torch.float32))
            trans.append(torch.tensor(camera.calibration.translation, dtype=torch.float32))
            intrins.append(torch.tensor(camera.intrinsic, dtype=torch.float32))
            post_rots.append(post_rot.float())
            post_trans.append(post_tran.float())
        return [torch.stack(t).unsqueeze(0).to(self.device) for t in (imgs, rots, trans, intrins, post_rots, post_trans)]

    @torch.inference_mode()
    def infer(self, frame: SurroundFrame) -> np.ndarray:
        """Vehicle probability on `LSS_GRID` in our BEV convention (row 0 = forward edge)."""
        prob = self.model(*self._inputs(frame)).sigmoid()[0, 0].cpu().numpy()
        # LSS indexes (x, y) from (x_min, y_min): flip both axes to put +x up and +y left.
        return np.flip(prob, axis=(0, 1)).copy()


def _blues(prob: np.ndarray) -> np.ndarray:
    """matplotlib-Blues lookalike, white at 0 to deep blue at 1, BGR."""
    t = np.clip(prob, 0.0, 1.0)[..., None]
    return (np.array([255, 255, 255], np.float32) * (1 - t) + np.array([130, 50, 8], np.float32) * t).astype(np.uint8)


def render_lss_bev(
    frame: SurroundFrame,
    prob: np.ndarray,
    view: BevGrid,
    map_renderer: NuScenesMapRenderer | None = None,
    threshold: float = 0.4,
) -> np.ndarray:
    """Paper-style render: Blues vehicle probability over the HD map."""
    image = map_renderer.render(frame, view) if map_renderer else canvas(view, (248, 248, 248))
    draw_metric_grid(image, view, dark=False)
    up = resample(prob.astype(np.float32), LSS_GRID, view)
    image = blend(image, _blues(up), np.clip(up * 1.4, 0.0, 1.0))
    draw_ego(image, view, fill=(118, 185, 0), outline=(60, 100, 0))
    return with_header(
        image,
        f"{frame.scene_name} | sample {frame.sample_index} | LSS camera-only BEV (vehicle seg)",
        f"mean={float(prob.mean()):.3f} | max={float(prob.max()):.3f} | thr={threshold:.2f}",
        dark=False,
    )


def overlay_lss(image: np.ndarray, prob: np.ndarray, view: BevGrid, threshold: float = 0.35, alpha: float = 0.75) -> np.ndarray:
    """Blend the LSS vehicle mask (cyan) onto a BEV image rendered on `view`."""
    weight = np.clip((resample(prob.astype(np.float32), LSS_GRID, view) - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
    return blend(image, (255, 180, 80), weight * alpha)
