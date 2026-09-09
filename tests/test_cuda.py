from depth_estimation_models import DepthModelBuilder


def test_get_device():
    builder = DepthModelBuilder("depth-anything/Depth-Anything-V2-Base-hf")

    builder._get_device()
    print(builder.device)

    assert builder.device in ("cuda", "cpu")
    assert builder.sysname
