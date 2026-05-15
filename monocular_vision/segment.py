import os
import time
import urllib.request
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from monocular_vision.logging_utils import get_logger

_WEIGHTS_URL = "https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt"
_WEIGHTS_PATH = "yolopv2.pt"

# Model was JIT-traced on 1280x720 source, letterboxed to 640x384
# (12px top/bottom padding from letterbox stride=32, auto=True)
_MODEL_W, _MODEL_H = 640, 384
_PAD_TOP = 12


class DrivableAreaSegmentor:
    def __init__(self):
        self.logger = get_logger("DrivableAreaSegmentor")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.use_half = self.device.startswith("cuda")
        self.model = torch.jit.load(self._ensure_weights(), map_location=self.device)
        self.model.eval()
        if self.use_half:
            self.model.half()
        self._warmup()
        self.logger.info(f"DrivableAreaSegmentor loaded on {self.device}.")

    def _ensure_weights(self):
        if not os.path.exists(_WEIGHTS_PATH):
            self.logger.info("Downloading YOLOPv2 weights from GitHub releases...")
            urllib.request.urlretrieve(_WEIGHTS_URL, _WEIGHTS_PATH)
            self.logger.info("YOLOPv2 weights downloaded.")
        return _WEIGHTS_PATH

    def _warmup(self):
        dummy = torch.zeros(1, 3, _MODEL_H, _MODEL_W, device=self.device)
        if self.use_half:
            dummy = dummy.half()
        with torch.no_grad():
            self.model(dummy)

    def _preprocess(self, frame):
        # Normalize to 1280x720 (trace assumption), then letterbox to 640x384
        img = cv2.resize(frame, (1280, 720))
        img = cv2.resize(img, (_MODEL_W, _MODEL_H - 2 * _PAD_TOP))  # 640x360
        img = cv2.copyMakeBorder(img, _PAD_TOP, _PAD_TOP, 0, 0,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
        tensor = torch.from_numpy(np.ascontiguousarray(img)).to(self.device).unsqueeze(0)
        tensor = tensor.half() if self.use_half else tensor.float()
        tensor /= 255.0
        return tensor

    def segment(self, frame):
        """Return (mask, elapsed_s). mask is uint8 at frame resolution, 1=drivable 0=not."""
        h, w = frame.shape[:2]
        t = time.time()
        tensor = self._preprocess(frame)
        with torch.no_grad():
            outputs = self.model(tensor)
        seg = outputs[1]
        # Strip letterbox padding rows, upsample 2x → 720x1280 matching source dims
        crop_end = _MODEL_H - _PAD_TOP
        da = seg[:, :, _PAD_TOP:crop_end, :]
        da = F.interpolate(da, scale_factor=2, mode='bilinear', align_corners=False)
        mask = torch.max(da, dim=1)[1].squeeze().cpu().numpy().astype(np.uint8)
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask, time.time() - t
