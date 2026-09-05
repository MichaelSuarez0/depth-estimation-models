import platform
import time
from typing import Any, Literal, Self

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from depth_estimation_models.utils.models import CMAPS, Models


class ModelBuilder:
    def __init__(self, model_id: Models) -> None:
        self.model_id = model_id
        self.device: Literal["cuda", "cpu"]

        self.cap: cv2.VideoCapture
        self.CMAPS = CMAPS

    def _get_device(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sysname = platform.system().lower()

    def build_model(self, half: bool = False) -> Self:
        self._get_device()
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

    def to_depth(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb)
        inputs = self.image_processor(images=pil, return_tensors="pt").to(self.device)
        with torch.no_grad(), torch.amp.autocast(enabled=True, device_type="cuda"):
            outputs = self.model(**inputs)
        post = self.image_processor.post_process_depth_estimation(
            outputs, target_sizes=[(pil.height, pil.width)]
        )[0]["predicted_depth"]
        depth = (post - post.min()) / (post.max() - post.min() + 1e-8)
        depth_np = (depth.detach().cpu().numpy() * 255.0).astype(np.uint8)
        return depth_np

    @staticmethod
    def overlay_text(img, text, pos, scale=0.7, color=(255, 255, 255)):
        cv2.putText(
            img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA
        )

    @staticmethod
    def colorize(depth_gray, cmap: dict[str, Any], cmap_name: str):
        cmap = cmap.get(cmap_name, "turbo")
        return cv2.applyColorMap(depth_gray, cmap)

    @staticmethod
    def compose_side_by_side(frame, depth_color):
        h, w = frame.shape[:2]
        depth_color = cv2.resize(depth_color, (w, h), interpolation=cv2.INTER_NEAREST)
        return np.hstack([frame, depth_color])

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

    # PIPELINE FUNCTIONS
    def run_video_capture(
        self, cmap_name: str, hflip: bool = False, max_fps: float = 0
    ):
        ema_fps, t_prev = None, time.time()

        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            if hflip:
                frame = cv2.flip(frame, 1)

            if max_fps > 0:
                t_now = time.time()
                dt = t_now - t_prev
                if dt < 1.0 / max_fps:
                    time.sleep((1.0 / max_fps) - dt)
                t_prev = time.time()

            # frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            t0 = time.time()
            dmap = self.to_depth(frame)
            dcol = self.colorize(dmap, self.CMAPS, cmap_name)

            fused = self.compose_side_by_side(frame, dcol)
            t1 = time.time()

            fps = 1.0 / max(t1 - t0, 1e-6)
            ema_fps = fps if ema_fps is None else 0.9 * ema_fps + 0.1 * fps
            w = fused.shape[1] // 2
            self.overlay_text(fused, "RGB", (15, 30))
            self.overlay_text(fused, self.model_id, (w + 15, 30))
            self.overlay_text(
                fused,
                f"FPS: {ema_fps:.1f}",
                (15, fused.shape[0] - 20),
                0.8,
                (20, 220, 20),
            )

            cv2.imshow(
                "Depth Anything V2 | Izquierda: RGB  |  Derecha: Profundidad", fused
            )
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()
