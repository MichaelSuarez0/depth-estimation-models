from typing import Literal

import cv2

Models = Literal[
    "depth-anything/Depth-Anything-V2-Base-hf",
    "Intel/zoedepth-kitti",
    "depth-anything/Depth-Anything-V2-Small-hf",
    "apple/DepthPro-hf",
]

CMAPS = {
    "turbo": cv2.COLORMAP_TURBO,
    "jet": cv2.COLORMAP_JET,
    "inferno": cv2.COLORMAP_INFERNO,
    "plasma": cv2.COLORMAP_PLASMA,
    "magma": cv2.COLORMAP_MAGMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "hot": cv2.COLORMAP_HOT,
    "cool": cv2.COLORMAP_COOL,
    "ocean": cv2.COLORMAP_OCEAN,
}
