import json
import math

import numpy as np

from conftest import make_frame, yaw_pose
from fsd.bev import BevGrid
from fsd.nuscenes_map import MAP_LAYERS, NuScenesMapRenderer


def _square(cx, cy, half=1.0):
    return [(cx - half, cy - half), (cx + half, cy - half), (cx + half, cy + half), (cx - half, cy + half)]


def _write_map(root, drivable_center, lane_center=None, divider=None):
    nodes, polygons, lines = [], [], []

    def polygon(token, corners):
        ids = []
        for x, y in corners:
            ids.append(f"n{len(nodes)}")
            nodes.append({"token": ids[-1], "x": x, "y": y})
        polygons.append({"token": token, "exterior_node_tokens": ids, "holes": []})

    polygon("drivable", _square(*drivable_center))
    data = {"drivable_area": [{"polygon_tokens": ["drivable"]}], "walkway": [], "ped_crossing": [], "road_divider": [], "lane_divider": []}
    if lane_center is not None:
        polygon("lane", _square(*lane_center))
        data["lane"] = [{"polygon_token": "lane"}]
    if divider is not None:
        ids = []
        for x, y in divider:
            ids.append(f"n{len(nodes)}")
            nodes.append({"token": ids[-1], "x": x, "y": y})
        lines.append({"token": "div", "node_tokens": ids})
        data["lane_divider"] = [{"line_token": "div"}]
    data.update(node=nodes, polygon=polygons, line=lines)
    (root / "maps" / "expansion").mkdir(parents=True)
    (root / "maps" / "expansion" / "boston-seaport.json").write_text(json.dumps(data))


def _cell(grid, x, y):
    rows, cols, inside = grid.cells(np.array([[x, y]]))
    assert inside[0]
    return rows[0], cols[0]


def test_features_in_the_view_corners_are_drawn_when_rotated(tmp_path):
    # Ego yaw 45 deg: the view corner (49, 49) in the ego frame is ~69 m straight
    # along global +y, outside the old 60 m culling radius.
    _write_map(tmp_path, (0.0, 49 * math.sqrt(2)))
    grid = BevGrid(resolution=0.5)
    image = NuScenesMapRenderer(tmp_path).render(make_frame(yaw_pose(yaw=math.pi / 4)), grid)
    assert tuple(image[_cell(grid, 49.0, 49.0)]) != (248, 248, 248)


def test_layer_masks_rasterise_each_layer_in_its_own_channel(tmp_path):
    _write_map(tmp_path, (10.0, 5.0), lane_center=(-10.0, -5.0), divider=[(20.0, -10.0), (20.0, 10.0)])
    grid = BevGrid(resolution=0.25)
    masks = NuScenesMapRenderer(tmp_path).layer_masks(make_frame(yaw_pose()), grid)
    assert masks.shape == (len(MAP_LAYERS), *grid.shape) and masks.dtype == np.uint8
    drivable = masks[:, _cell(grid, 10.0, 5.0)[0], _cell(grid, 10.0, 5.0)[1]]
    assert drivable.tolist() == [1, 0, 0, 0, 0, 0, 0, 0]
    assert masks[1][_cell(grid, -10.0, -5.0)] == 1
    assert masks[MAP_LAYERS.index("lane_divider")][_cell(grid, 20.0, 0.0)] == 1
