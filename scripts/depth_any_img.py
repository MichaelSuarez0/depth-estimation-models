from pathlib import Path

from depth_estimation_models import DepthModelBuilder

ROOT = Path(__file__).parent
OUTPUT_PATH = ROOT / "output"
OUTPUT_PATH.mkdir(exist_ok=True)

IMAGE_PATH = ROOT / "img1.jpeg"
OUT_GRAY = OUTPUT_PATH / "depth_gray_img1b.png"
OUT_COLOR = OUTPUT_PATH / "depth_color_img1b.png"


def main():
    mb = DepthModelBuilder(model_id="depth-anything/Depth-Anything-V2-Base-hf")
    (
        mb.build_model()
        .set_normalization("relative")
        .infer_depth_from_img(IMAGE_PATH, OUT_GRAY, OUT_COLOR)
    )


if __name__ == "__main__":
    main()
