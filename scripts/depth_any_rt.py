from depth_estimation_models import DepthModelBuilder


def main():
    mb = DepthModelBuilder(model_id="depth-anything/Depth-Anything-V2-Small-hf")
    (
        mb.build_model(half=True)
        .set_normalization("relative")
        .set_video_capture()
        .run_video_capture(cmap_name="inferno")
    )


if __name__ == "__main__":
    main()
