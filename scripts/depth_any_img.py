from pathlib import Path

from depth_estimation_models import ModelBuilder

ROOT = Path(__file__).parent
OUTPUT_PATH = ROOT / "output"
IMAGE_PATH = ROOT / "img1.jpeg"
OUT_GRAY = OUTPUT_PATH / "depth_gray_img1b.png"
OUT_COLOR = OUTPUT_PATH / "depth_color_img1b.png"

OUTPUT_PATH.mkdir(exist_ok=True)


def main():
    mb = ModelBuilder(model_id="depth-anything/Depth-Anything-V2-Base-hf")
    mb.build_model().infer_depth_from_img(IMAGE_PATH, OUT_GRAY, OUT_COLOR)


if __name__ == "__main__":
    main()
