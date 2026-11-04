from pathlib import Path

import pytest

from fsd.visualize import LSS_VIEWS, VIEWS, _output_paths, _select_views, parse_args


def test_all_skips_lss_views_without_weights():
    assert set(_select_views("all", None)) == set(VIEWS) - LSS_VIEWS
    assert _select_views("all", "lss.pt") == list(VIEWS)
    with pytest.raises(ValueError, match="--lss-weights"):
        _select_views("lss_bev", None)


def test_output_paths_suffix_a_shared_base_per_view():
    assert _output_paths("out/run.mp4", ["bev"]) == {"bev": Path("out/run.mp4")}
    assert _output_paths("out/run.mp4", ["bev", "lidar"])["lidar"] == Path("out/run_lidar.mp4")


def test_scene_range_and_frames_args():
    opts = parse_args(["--scenes", "2-4", "--frames", "all", "--save"])
    assert opts.scenes == [2, 3, 4] and opts.frames is None and not opts.stream
