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

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from efficientnet_pytorch import EfficientNet
from torch import nn
from torchvision.models.resnet import resnet18

from fsd.data import CAMERA_CHANNELS, SurroundFrame
from fsd.lidar_projection import quaternion_to_rotation_matrix


# nuScenes camera order LSS was trained on (same as ours).
LSS_CAMS = list(CAMERA_CHANNELS)

DEFAULT_GRID_CONF = {
    "xbound": [-50.0, 50.0, 0.5],
    "ybound": [-50.0, 50.0, 0.5],
    "zbound": [-10.0, 10.0, 20.0],
    "dbound": [4.0, 45.0, 1.0],
}
DEFAULT_DATA_AUG_CONF = {
    "resize_lim": (0.193, 0.225),
    "final_dim": (128, 352),
    "rot_lim": (-5.4, 5.4),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
    "bot_pct_lim": (0.0, 0.22),
    "cams": LSS_CAMS,
    "Ncams": 5,
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = ranks[1:] != ranks[:-1]
    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))
    return x, geom_feats


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
    def __init__(self, D, C, downsample, pretrained_trunk: bool = False):
        super().__init__()
        self.D = D
        self.C = C
        # Skip downloading EfficientNet ImageNet weights when we plan to load
        # a full LSS checkpoint right after.
        if pretrained_trunk:
            self.trunk = EfficientNet.from_pretrained("efficientnet-b0")
        else:
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
    def __init__(self, grid_conf, data_aug_conf, outC, pretrained_trunk: bool = False):
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
        self.camencode = CamEncode(self.D, self.camC, self.downsample, pretrained_trunk=pretrained_trunk)
        self.bevencode = BevEncode(inC=self.camC, outC=outC)
        self.use_quickcumsum = True

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
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
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

    def __init__(
        self,
        weights_path: str | Path,
        device: str | torch.device | None = None,
        grid_conf: dict | None = None,
        data_aug_conf: dict | None = None,
        out_channels: int = 1,
    ):
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"LSS weights not found: {self.weights_path}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.grid_conf = grid_conf or DEFAULT_GRID_CONF
        self.data_aug_conf = data_aug_conf or DEFAULT_DATA_AUG_CONF

        self.model = LiftSplatShoot(self.grid_conf, self.data_aug_conf, outC=out_channels)
        state = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        nx = self.model.nx.detach().cpu().numpy()
        self.bev_x_cells = int(nx[0])
        self.bev_y_cells = int(nx[1])

    def _per_camera_tensors(self, frame: SurroundFrame):
        imgs, rots, trans, intrins, post_rots, post_trans = [], [], [], [], [], []
        for channel in LSS_CAMS:
            camera = frame.cameras[channel]
            image = Image.open(camera.image_path).convert("RGB")
            image, post_rot, post_tran = _eval_image_transform(image, self.data_aug_conf)

            intrinsic = torch.tensor(camera.camera_intrinsic, dtype=torch.float32)
            rot = torch.tensor(
                quaternion_to_rotation_matrix(camera.calibrated_sensor["rotation"]),
                dtype=torch.float32,
            )
            tran = torch.tensor(camera.calibrated_sensor["translation"], dtype=torch.float32)

            imgs.append(_normalize_image(image))
            intrins.append(intrinsic)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot.float())
            post_trans.append(post_tran.float())

        return (
            torch.stack(imgs).unsqueeze(0),
            torch.stack(rots).unsqueeze(0),
            torch.stack(trans).unsqueeze(0),
            torch.stack(intrins).unsqueeze(0),
            torch.stack(post_rots).unsqueeze(0),
            torch.stack(post_trans).unsqueeze(0),
        )

    @torch.inference_mode()
    def infer(self, frame: SurroundFrame) -> np.ndarray:
        """Return a (X, Y) numpy array of sigmoided vehicle-segmentation probs.

        Indexing follows LSS's grid: index 0 in X corresponds to xbound[0]
        (rear of ego), index 0 in Y to ybound[0] (right of ego).
        """
        imgs, rots, trans, intrins, post_rots, post_trans = self._per_camera_tensors(frame)
        out = self.model(
            imgs.to(self.device),
            rots.to(self.device),
            trans.to(self.device),
            intrins.to(self.device),
            post_rots.to(self.device),
            post_trans.to(self.device),
        )
        return out.sigmoid()[0, 0].detach().cpu().numpy()


def _lss_grid_to_bev_pixels(prob_map: np.ndarray) -> np.ndarray:
    """Reorient a (X, Y) LSS grid so row=0 is the top (forward) of our BEV."""
    return np.flip(prob_map, axis=(0, 1)).copy()


def _blues_colormap(prob: np.ndarray) -> np.ndarray:
    """matplotlib-Blues lookalike — white at 0, deep blue at 1, BGR uint8."""
    t = np.clip(prob, 0.0, 1.0)[..., None]
    white = np.array([255, 255, 255], dtype=np.float32)
    deep_blue = np.array([130, 50, 8], dtype=np.float32)  # BGR
    rgb = white * (1 - t) + deep_blue * t
    return rgb.astype(np.uint8)


def _draw_metric_grid(canvas: np.ndarray, x_range, y_range, resolution, dark: bool) -> None:
    x_min, x_max = x_range
    y_min, y_max = y_range
    h, w = canvas.shape[:2]
    if dark:
        minor, major = (60, 60, 60), (110, 110, 110)
    else:
        minor, major = (220, 220, 220), (170, 170, 170)
    for x in range(int(np.ceil(x_min / 10) * 10), int(x_max) + 1, 10):
        row = int((x_max - x) / resolution)
        if 0 <= row < h:
            cv2.line(canvas, (0, row), (w - 1, row), major if x == 0 else minor, 1)
            cv2.putText(canvas, f"{x}m", (6, max(14, row - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, major if x == 0 else minor, 1)
    for y in range(int(np.ceil(y_min / 10) * 10), int(y_max) + 1, 10):
        col = int((y_max - y) / resolution)
        if 0 <= col < w:
            cv2.line(canvas, (col, 0), (col, h - 1), major if y == 0 else minor, 1)
            cv2.putText(canvas, f"{y}m", (max(2, col + 3), h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, major if y == 0 else minor, 1)


def _draw_ego_marker(canvas: np.ndarray, x_range, y_range, resolution, color, outline) -> None:
    _, x_max = x_range
    _, y_max = y_range
    row = int(x_max / resolution)
    col = int(y_max / resolution)
    car_w = max(6, int(2.0 / resolution))
    car_l = max(12, int(4.5 / resolution))
    cv2.rectangle(canvas, (col - car_w // 2, row - car_l // 2), (col + car_w // 2, row + car_l // 2), color, -1)
    cv2.rectangle(canvas, (col - car_w // 2, row - car_l // 2), (col + car_w // 2, row + car_l // 2), outline, 1)
    cv2.arrowedLine(canvas, (col, row), (col, row - car_l), outline, 2, tipLength=0.35)


def render_lss_bev(
    frame: SurroundFrame,
    prob_map: np.ndarray,
    grid_conf: dict | None = None,
    scale: int = 4,
    threshold: float = 0.4,
    style: str = "paper",
    map_background: np.ndarray | None = None,
) -> np.ndarray:
    """Render the LSS BEV vehicle-segmentation output.

    `style="paper"` mirrors the original LSS visualization (white background,
    Blues colormap). `style="dark"` blends the prob map onto a dark BEV canvas
    matching `render_lidar_bev`. Pass `map_background` to use a nuScenes HD-map
    render as the backdrop (like the original LSS paper figure).
    """
    grid_conf = grid_conf or DEFAULT_GRID_CONF
    x_min, x_max, x_res = grid_conf["xbound"]
    y_min, y_max, y_res = grid_conf["ybound"]

    grid = _lss_grid_to_bev_pixels(prob_map).astype(np.float32)
    native_h, native_w = grid.shape
    out_h, out_w = native_h * scale, native_w * scale
    grid_up = cv2.resize(grid, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    if map_background is not None:
        if map_background.shape[:2] != (out_h, out_w):
            map_background = cv2.resize(map_background, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        canvas = map_background.copy()
        dark = False
    else:
        dark = style != "paper"
        bg = 22 if dark else 248
        canvas = np.full((out_h, out_w, 3), bg, dtype=np.uint8)
    _draw_metric_grid(canvas, (x_min, x_max), (y_min, y_max), x_res / scale, dark=dark)

    if style == "paper" or map_background is not None:
        heat = _blues_colormap(grid_up)
        alpha = np.clip(grid_up * 1.4, 0.0, 1.0)[..., None]
        canvas = (canvas.astype(np.float32) * (1 - alpha) + heat.astype(np.float32) * alpha).astype(np.uint8)
    else:
        heat = cv2.applyColorMap((np.clip(grid_up, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLORMAP_OCEAN)
        alpha = np.clip((grid_up - 0.05) / 0.95, 0.0, 1.0)[..., None]
        canvas = (canvas.astype(np.float32) * (1 - alpha) + heat.astype(np.float32) * alpha).astype(np.uint8)

    ego_fill = (118, 185, 0) if (style == "paper" or map_background is not None) else (220, 220, 240)
    ego_outline = (60, 100, 0) if (style == "paper" or map_background is not None) else (90, 90, 130)
    _draw_ego_marker(canvas, (x_min, x_max), (y_min, y_max), x_res / scale, ego_fill, ego_outline)

    header_h = 58
    paper_like = style == "paper" or map_background is not None
    text_color = (40, 40, 40) if paper_like else (245, 245, 245)
    sub_color = (80, 80, 150) if paper_like else (190, 220, 255)
    header_bg = 240 if paper_like else 18
    output = np.full((header_h + out_h, out_w, 3), header_bg, dtype=np.uint8)
    output[header_h:, :] = canvas
    title = f"{frame.scene_name} | sample {frame.sample_index} | LSS BEV (vehicle seg)"
    stats = f"mean={float(prob_map.mean()):.3f} | max={float(prob_map.max()):.3f} | thr={threshold:.2f}"
    cv2.putText(output, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, text_color, 1, cv2.LINE_AA)
    cv2.putText(output, stats, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.46, sub_color, 1, cv2.LINE_AA)
    return output


def overlay_lss_on_lidar_bev(
    lidar_bev_image: np.ndarray,
    prob_map: np.ndarray,
    lidar_resolution: float,
    lidar_scale: int,
    grid_conf: dict | None = None,
    header_h: int = 58,
    threshold: float = 0.35,
    alpha: float = 0.75,
) -> np.ndarray:
    """Blend the LSS vehicle BEV mask on top of an existing LiDAR BEV render.

    `lidar_resolution` and `lidar_scale` must match the values used to render
    `lidar_bev_image` — they control how the 200x200 LSS grid is resampled
    onto the LiDAR canvas. Without these the overlay drifts (the bug in the
    first cut of this view).
    """
    grid_conf = grid_conf or DEFAULT_GRID_CONF
    x_min, x_max, _ = grid_conf["xbound"]
    y_min, y_max, _ = grid_conf["ybound"]

    lidar_grid_h = int(round((x_max - x_min) / lidar_resolution)) * lidar_scale
    lidar_grid_w = int(round((y_max - y_min) / lidar_resolution)) * lidar_scale
    scaled_header = header_h * lidar_scale

    grid = _lss_grid_to_bev_pixels(prob_map).astype(np.float32)
    grid = cv2.resize(grid, (lidar_grid_w, lidar_grid_h), interpolation=cv2.INTER_LINEAR)

    out = lidar_bev_image.copy()
    band = out[scaled_header : scaled_header + lidar_grid_h, :lidar_grid_w]
    if band.shape[:2] != grid.shape:
        return out

    weight = np.clip((grid - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0) * alpha
    weight = weight[..., None]
    overlay_color = np.array([255, 180, 80], dtype=np.float32)  # BGR — bright cyan-blue
    blended = band.astype(np.float32) * (1 - weight) + overlay_color * weight
    out[scaled_header : scaled_header + lidar_grid_h, :lidar_grid_w] = blended.astype(np.uint8)

    cv2.putText(
        out,
        "LSS vehicle seg (cyan)",
        (10, scaled_header + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 220, 180),
        1,
        cv2.LINE_AA,
    )
    return out
