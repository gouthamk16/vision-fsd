from pathlib import Path

import pytest

from conftest import yaw_pose
from fsd.bev import BevGrid
from fsd.data import LidarFrame, LidarSweep
from fsd.visualize import LSS_VIEWS, VIEWS, _Context, _output_paths, _Products, _select_views, parse_args


def test_all_skips_lss_views_without_weights():
    assert set(_select_views("all", None, "nmp.pt")) == set(VIEWS) - LSS_VIEWS
    assert _select_views("all", "lss.pt", "nmp.pt") == list(VIEWS)
    with pytest.raises(ValueError, match="--lss-weights"):
        _select_views("lss_bev", None, None)


def test_all_skips_planner_without_weights():
    assert "planner_bev" not in _select_views("all", "lss.pt", None)
    assert "planner_bev" in VIEWS
    with pytest.raises(ValueError, match="--nmp-weights"):
        _select_views("planner_bev", None, None)


def test_output_paths_suffix_a_shared_base_per_view():
    assert _output_paths("out/run.mp4", ["bev"]) == {"bev": Path("out/run.mp4")}
    assert _output_paths("out/run.mp4", ["bev", "lidar"])["lidar"] == Path("out/run_lidar.mp4")


def test_scene_range_and_frames_args():
    opts = parse_args(["--scenes", "2-4", "--frames", "all", "--save"])
    assert opts.scenes == [2, 3, 4] and opts.frames is None and not opts.stream


def test_views_draw_only_the_requested_sweeps_of_a_longer_planner_history():
    history = LidarFrame(tuple(LidarSweep(Path(f"{i}.bin"), -i, yaw_pose(), yaw_pose()) for i in range(19)))
    ctx = _Context(BevGrid(), BevGrid(), 640, 80.0, 1, 0.5, lidar_sweeps=2)
    assert len(_Products(None, history, ctx).view_lidar.sweeps) == 2
