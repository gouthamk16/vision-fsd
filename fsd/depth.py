import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from fsd.logging_utils import get_logger


class MonocularDepthEstimator:
    def __init__(self, model_name="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"):
        self.logger = get_logger("MonocularDepthEstimator")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_name)
        self.model.to(self.device).eval()
        self.logger.info(f"Depth estimator initialized with {self.model_name} on {self.device}.")

    def estimate(self, frame):
        start_time = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = F.interpolate(
                outputs.predicted_depth.unsqueeze(1),
                size=frame.shape[:2],
                mode="bilinear",
                align_corners=True,
            )

        depth_map = prediction.squeeze().detach().cpu().numpy().astype(np.float32)
        depth_map = np.maximum(depth_map, 0.0)
        return depth_map, time.time() - start_time

