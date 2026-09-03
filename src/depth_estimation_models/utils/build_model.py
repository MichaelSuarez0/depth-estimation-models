import platform
from typing import Literal, Self

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class ModelBuilder:
    def __init__(self, model_id) -> None:
        self.model_id = model_id
        self.device: Literal["cuda", "cpu"]

        self.cap: cv2.VideoCapture

    def _get_device(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sysname = platform.system().lower()

    def build_model(self, half) -> Self:
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
        self.model.to(self.device)
        if half:
            self.model.half()
        self.model.eval()
        return self

    def set_video_capture(
        self, source: int = 0, width: int = 640, height: int = 480
    ) -> Self:
        backends = []
        if "windows" in self.sysname:
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        elif "linux" in self.sysname:
            backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]

        cap = None
        for be in backends:
            cap = cv2.VideoCapture(source, be)
            if cap.isOpened():
                break
            if cap is not None:
                cap.release()
                cap = None
        if cap and cap.isOpened():
            self.cap = cap
        else:
            raise ValueError(
                "No se pudo abrir la cámara. Probá con SOURCE=1 o verificá permisos/backend de OpenCV."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return self

    def to_depth(self, frame_bgr, device):

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb)
        inputs = self.image_processor(images=pil, return_tensors="pt").to(device)
        with torch.no_grad(), torch.amp.autocast(enabled=True, device_type="cuda"):
            outputs = self.model(**inputs)
        post = self.image_processor.post_process_depth_estimation(
            outputs, target_sizes=[(pil.height, pil.width)]
        )[0]["predicted_depth"]
        depth = (post - post.min()) / (post.max() - post.min() + 1e-8)
        depth_np = (depth.detach().cpu().numpy() * 255.0).astype(np.uint8)
        return depth_np

    def infer_depth_meter(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb)
        inputs = self.image_processor(images=pil, return_tensors="pt").to(self.device)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            outputs = self.model(**inputs)
        post = self.image_processor.post_process_depth_estimation(
            outputs, target_sizes=[(pil.height, pil.width)]
        )[0]["predicted_depth"]
        return post.float()
