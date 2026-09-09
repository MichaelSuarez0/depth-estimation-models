from pathlib import Path

from depth_estimation_models import DepthModelBuilder

ROOT = Path(__file__).parent
OUTPUT_PATH = ROOT / "output"
IMAGE_PATH = ROOT / "img1.jpeg"
OUT_GRAY = OUTPUT_PATH / "depth_gray_img1b.png"
OUT_COLOR = OUTPUT_PATH / "depth_color_img1b.png"

OUTPUT_PATH.mkdir(exist_ok=True)


def main():
    mb = DepthModelBuilder(model_id="apple/DepthPro-hf")
    (
        mb.build_model(half=True)
        .set_normalization("absolute", vmin=0.2, vmax=10.0)
        .set_video_capture()
        .run_video_capture(cmap_name="turbo")
    )


if __name__ == "__main__":
    main()
