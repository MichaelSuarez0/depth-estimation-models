import platform
import time
from pathlib import Path
from typing import Any, Literal, Self

import cv2
import numpy as np
import PIL.Image
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from depth_estimation_models.core.models import CMAPS, Models


class DepthModelBuilder:
    def __init__(
        self,
        model_id: Models,
    ) -> None:
        self.model_id = model_id
        self.device: Literal["cuda", "cpu"]
        self.sysname: str

        self.cap: cv2.VideoCapture
        self.width: int
        self.height: int
        self.CMAPS = CMAPS

    def set_normalization(
        self,
        mode: Literal["relative", "absolute"],
        vmin: float = 0.2,
        vmax: float = 10.0,
    ) -> Self:
        """Cambia la config de normalización después de construir la instancia."""
        self.mode = mode
        self.vmin = vmin
        self.vmax = vmax
        return self

    # ---------------------------------------------------------------
    # SETUP
    # ---------------------------------------------------------------
    def _get_device(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Utilizando {self.device}")
        self.sysname = platform.system().lower()

    def build_model(self, half: bool = False) -> Self:
        self._get_device()
        self.half = half and (self.device == "cuda")
        torch.backends.cudnn.benchmark = True

        self.image_processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
        self.model.to(self.device)
        if self.half:
            self.model.half()
        self.model.eval()
        return self

    def set_video_capture(
        self, source: int = 0, width: int = 640, height: int = 480
    ) -> Self:
        self.width, self.height = width, height

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

    # ---------------------------------------------------------------
    # HELPERS (estáticos, sin estado)
    # ---------------------------------------------------------------
    @staticmethod
    def add_overlay_text(img, text, pos, scale=0.7, color=(255, 255, 255)):
        cv2.putText(
            img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA
        )

    @staticmethod
    def colorize(depth_gray: np.ndarray, cmap: dict[str, Any], cmap_name: str):
        cv_cmap = cmap.get(cmap_name, cv2.COLORMAP_TURBO)
        return cv2.applyColorMap(depth_gray, cv_cmap)

    @staticmethod
    def compose_side_by_side(frame, depth_color):
        h, w = frame.shape[:2]
        depth_color = cv2.resize(depth_color, (w, h), interpolation=cv2.INTER_NEAREST)
        return np.hstack([frame, depth_color])

    def _normalize_for_viz(self, depth: torch.Tensor) -> np.ndarray:
        """
        Convierte un tensor de profundidad (en metros, float) a una imagen
        uint8 de un canal (0-255) lista para colorear.

        Usa self.mode / self.vmin / self.vmax (ver set_normalization).
        mode="absolute": usa un rango fijo [vmin, vmax] -> comparable entre frames.
        mode="relative": usa el min/max de cada frame -> máximo contraste local.
        """
        if self.mode == "absolute":
            d = depth.clamp(self.vmin, self.vmax)
            d = (d - self.vmin) / max(self.vmax - self.vmin, 1e-8)
        else:
            dmin = depth.min()
            dmax = depth.max()
            d = (depth - dmin) / (dmax - dmin + 1e-8)

        return (d.detach().cpu().numpy() * 255.0).astype(np.uint8)

    # ---------------------------------------------------------------
    # INFERENCIA
    # ---------------------------------------------------------------
    def _run_inference(self, image: PIL.Image.Image) -> np.ndarray:
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
        with (
            torch.no_grad(),
            torch.amp.autocast("cuda", enabled=(self.device == "cuda")),
        ):
            outputs = self.model(**inputs)

        post = self.image_processor.post_process_depth_estimation(
            outputs, target_sizes=[(image.height, image.width)]
        )[0]["predicted_depth"].float()

        return self._normalize_for_viz(post)

    def _infer_depth_meter(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(frame_rgb)
        return self._run_inference(image)

    def infer_depth_from_img(
        self,
        image_path: Path,
        out_gray: Path,
        out_color: Path,
        cmap_name: str = "turbo",
    ) -> Self:
        image = PIL.Image.open(image_path).convert("RGB")
        depth_gray = self._run_inference(image)
        depth_color = self.colorize(depth_gray, self.CMAPS, cmap_name)

        cv2.imwrite(str(out_gray), depth_gray)
        cv2.imwrite(str(out_color), depth_color)

        cv2.imshow("Imagen original", cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        cv2.imshow("Profundidad (color)", depth_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return self

    # ---------------------------------------------------------------
    # PIPELINE DE VIDEO (genérico, reutilizable por cualquier modelo)
    # ---------------------------------------------------------------
    def run_video_capture(
        self,
        cmap_name: str = "turbo",
        window_name: str | None = None,
        hflip: bool = False,
        max_fps: float = 0.0,
        resize: bool = False,
        show_placeholder: bool = True,
    ) -> None:
        window_name = (
            window_name or f"{self.model_id} | Izquierda: RGB  |  Derecha: Profundidad"
        )

        if show_placeholder:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            placeholder = np.zeros((self.height, 2 * self.width, 3), dtype=np.uint8)
            self.add_overlay_text(placeholder, "Inicializando...", (20, 40))
            cv2.imshow(window_name, placeholder)
            cv2.waitKey(1)

        ema_fps, t_prev = None, time.time()

        while True:
            ok, frame = self.cap.read()
            if not ok:
                print("No se pudo leer frame de la cámara.")
                break

            if hflip:
                frame = cv2.flip(frame, 1)

            if max_fps > 0:
                t_now = time.time()
                dt = t_now - t_prev
                if dt < 1.0 / max_fps:
                    time.sleep((1.0 / max_fps) - dt)
                t_prev = time.time()

            if resize:
                frame = cv2.resize(
                    frame, (self.width, self.height), interpolation=cv2.INTER_AREA
                )

            t0 = time.time()
            depth_gray = self._infer_depth_meter(frame)
            depth_color = self.colorize(depth_gray, self.CMAPS, cmap_name)
            fused = self.compose_side_by_side(frame, depth_color)
            t1 = time.time()

            fps = 1.0 / max(t1 - t0, 1e-6)
            ema_fps = fps if ema_fps is None else 0.9 * ema_fps + 0.1 * fps

            w = fused.shape[1] // 2
            self.add_overlay_text(fused, "RGB", (15, 30))
            self.add_overlay_text(fused, f"{self.model_id} | {self.mode}", (w + 15, 30))
            self.add_overlay_text(
                fused,
                f"FPS: {ema_fps:.1f}",
                (15, fused.shape[0] - 20),
                0.8,
                (20, 220, 20),
            )

            cv2.imshow(window_name, fused)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()
