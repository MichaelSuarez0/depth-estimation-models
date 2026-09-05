from depth_estimation_models import ModelBuilder


def main():
    mb = ModelBuilder(model_id="depth-anything/Depth-Anything-V2-Small-hf")
    mb.build_model().set_video_capture().run_video_capture(cmap_name="inferno")


if __name__ == "__main__":
    main()
