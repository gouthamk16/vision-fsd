"""Generate the 360-vision technical whitepaper PDF.

Equations are rendered with matplotlib mathtext (no LaTeX install needed) and
the document is laid out with fpdf2 (pure-Python, no native deps). Run:

    .venv/Scripts/python.exe docs/build_whitepaper.py

Output: docs/360_vision_whitepaper.pdf
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF
from PIL import Image

DOCS = Path(__file__).resolve().parent
ASSETS = DOCS / "_whitepaper_assets"
ASSETS.mkdir(exist_ok=True)
OUT = DOCS / "360_vision_whitepaper.pdf"

FONTS = Path("C:/Windows/Fonts")

# ---- colors -----------------------------------------------------------------
INK = (28, 30, 34)
MUTED = (110, 116, 126)
ACCENT = (28, 92, 168)
ACCENT2 = (158, 64, 24)
CODE_BG = (244, 245, 247)
CODE_INK = (40, 44, 52)
RULE = (210, 214, 220)

CONTENT_W = 174.0  # A4 width 210 - 2*18 margin


# ---- equation rendering -----------------------------------------------------
_eq_cache: dict[str, Path] = {}


def eq_png(latex: str, fontsize: float = 12.5, dpi: int = 300) -> Path:
    key = f"{latex}|{fontsize}|{dpi}"
    if key in _eq_cache:
        return _eq_cache[key]
    path = ASSETS / f"eq_{abs(hash(key)) & 0xFFFFFFFF:08x}.png"
    fig = plt.figure()
    fig.patch.set_alpha(0.0)
    t = fig.text(0.0, 0.0, f"${latex}$", fontsize=fontsize, color="#1c1e22")
    fig.savefig(path, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    _eq_cache[key] = path
    return path


def diagram_pipeline() -> Path:
    path = ASSETS / "pipeline.png"
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 38)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.4, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, zorder=3)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#5a6068", lw=1.3), zorder=1)

    box(2, 22, 20, 10, "nuScenes\n6 cams + LiDAR", "#eaf2fb", "#1c5ca8")
    box(2, 4, 20, 10, "ego poses\n(SE(3))", "#f3eee9", "#9e4018")

    box(30, 26, 18, 8, "LiDAR BEV\nrasterizer", "#eef4ec", "#3f7a3f")
    box(30, 16, 18, 8, "2.5D height\ntensor", "#eef4ec", "#3f7a3f")
    box(30, 6, 18, 8, "temporal\noccupancy", "#eef4ec", "#3f7a3f")

    box(56, 26, 18, 8, "object\ndetection", "#fdf1e7", "#9e4018")
    box(56, 14, 18, 8, "collision\ngrid", "#eef4ec", "#3f7a3f")

    box(80, 19, 18, 10, "BEV world\nmodel", "#eaf2fb", "#1c5ca8")
    box(80, 4, 18, 10, "local\nplanner", "#f0ecf6", "#5a3a8a")

    for y in (30, 20, 10):
        arrow(22, 27 if y == 30 else (20 if y == 20 else 9), 30, y)
    arrow(48, 30, 56, 30)
    arrow(48, 20, 56, 18)
    arrow(48, 10, 56, 16)
    arrow(74, 30, 80, 25)
    arrow(74, 18, 80, 22)
    arrow(89, 19, 89, 14)
    fig.savefig(path, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return _autocrop(path)


def _autocrop(path: Path, pad: int = 10) -> Path:
    """Trim transparent borders so the reserved PDF height matches visible content."""
    im = Image.open(path)
    if im.mode != "RGBA":
        return path
    bbox = im.getbbox()
    if bbox is None:
        return path
    left = max(0, bbox[0] - pad)
    top = max(0, bbox[1] - pad)
    right = min(im.width, bbox[2] + pad)
    bottom = min(im.height, bbox[3] + pad)
    im.crop((left, top, right, bottom)).save(path)
    return path


def _save(fig, name):
    path = ASSETS / name
    fig.savefig(path, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return _autocrop(path)


def diagram_frames():
    import numpy as np
    fig, ax = plt.subplots(figsize=(8.4, 3.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis("off")

    def triad(ox, oy, ang, label, color, L=9):
        a = np.radians(ang)
        ax.annotate("", xy=(ox + L * np.cos(a), oy + L * np.sin(a)), xytext=(ox, oy),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.7))
        ax.annotate("", xy=(ox + L * np.cos(a + np.pi / 2), oy + L * np.sin(a + np.pi / 2)),
                    xytext=(ox, oy), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.7))
        ax.plot(ox, oy, "o", color=color, ms=3.5)
        ax.text(ox, oy - 4.5, label, ha="center", va="top", fontsize=9, color=color, weight="bold")

    triad(11, 13, 33, "global (map)", "#1c5ca8")
    triad(47, 17, 6, "ego (body)", "#3f7a3f")
    triad(82, 24, -16, "sensor", "#9e4018")

    ax.annotate("", xy=(56, 18), xytext=(76, 24),
                arrowprops=dict(arrowstyle="-|>", color="#5a6068", lw=1.4,
                                connectionstyle="arc3,rad=-0.18"))
    ax.text(67, 30, "calibrated_sensor\nR, t : sensor -> ego", ha="center", fontsize=7.6)
    ax.annotate("", xy=(20, 15), xytext=(43, 17),
                arrowprops=dict(arrowstyle="-|>", color="#5a6068", lw=1.4,
                                connectionstyle="arc3,rad=-0.18"))
    ax.text(31, 22, "ego_pose\nR, t : ego -> global", ha="center", fontsize=7.6)
    ax.text(50, 3, "cameras invert this chain (global -> ego -> sensor) to project into the image",
            ha="center", fontsize=7.4, style="italic", color="#6e747e")
    return _save(fig, "frames.png")


def diagram_bev_grid():
    import numpy as np
    fig, ax = plt.subplots(figsize=(4.7, 4.7))
    n = 10
    ax.set_xlim(-0.5, n + 1.5)
    ax.set_ylim(-0.5, n + 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    for i in range(n + 1):
        ax.plot([0, n], [i, i], color="#d6dae0", lw=0.8, zorder=1)
        ax.plot([i, i], [0, n], color="#d6dae0", lw=0.8, zorder=1)
    # ego at centre, forward = up
    ax.add_patch(plt.Rectangle((n / 2 - 0.35, n / 2 - 0.6), 0.7, 1.2,
                               facecolor="#3f7a3f", edgecolor="#26492a", zorder=4))
    ax.annotate("", xy=(n / 2, n / 2 + 2.0), xytext=(n / 2, n / 2),
                arrowprops=dict(arrowstyle="-|>", color="#26492a", lw=1.6), zorder=4)
    ax.text(n / 2 + 0.25, n / 2 + 1.9, "+x (forward)", fontsize=7.6, color="#26492a", va="center")
    ax.annotate("", xy=(n / 2 - 2.0, n / 2), xytext=(n / 2, n / 2),
                arrowprops=dict(arrowstyle="-|>", color="#26492a", lw=1.6), zorder=4)
    ax.text(n / 2 - 2.1, n / 2 + 0.4, "+y (left)", fontsize=7.6, color="#26492a", ha="center")
    # a sample point and its cell
    px, py = 7.5, 7.5
    ax.add_patch(plt.Rectangle((int(px), int(py)), 1, 1, facecolor="#f0b67a",
                               edgecolor="#9e4018", zorder=3))
    ax.plot(px, py, "o", color="#9e4018", ms=4, zorder=5)
    ax.text(px + 0.2, py + 0.25, "point (x, y)", fontsize=7.6, color="#9e4018")
    ax.annotate("row = (x_max - x) / d", xy=(int(px) + 0.5, int(py)), xytext=(n + 0.3, 3.2),
                fontsize=7.4, color="#9e4018",
                arrowprops=dict(arrowstyle="-", color="#9e4018", lw=0.8))
    ax.annotate("col = (y_max - y) / d", xy=(int(px), int(py) + 0.5), xytext=(n + 0.3, 2.2),
                fontsize=7.4, color="#9e4018",
                arrowprops=dict(arrowstyle="-", color="#9e4018", lw=0.8))
    ax.text(n / 2, n + 0.9, "metric window [x_min,x_max] x [y_min,y_max] at d m/cell",
            ha="center", fontsize=7.8, color="#3a3f47")
    ax.text(n / 2, -0.2, "row 0 = top (far forward),   col 0 = left",
            ha="center", va="top", fontsize=7.2, color="#6e747e")
    return _save(fig, "bev_grid.png")


def diagram_occupancy():
    import numpy as np
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    def grid(cx, cy, wall_row, title):
        s = 2.2
        m = 6
        for r in range(m):
            for c in range(m):
                col = "#9aa0a8"  # unknown gray
                if r == wall_row:
                    col = "#ffba3a"  # occupied
                elif r > wall_row:
                    col = "#243f5a"  # free corridor (navy) behind nothing... below wall toward ego
                ax.add_patch(plt.Rectangle((cx + c * s, cy + r * s), s * 0.94, s * 0.94,
                                           facecolor=col, edgecolor="none"))
        ax.text(cx + m * s / 2, cy - 2.4, title, ha="center", fontsize=8, color="#3a3f47")
        # ego marker bottom centre
        ax.add_patch(plt.Rectangle((cx + m * s / 2 - 1.0, cy - 0.2), 2.0, 1.6,
                                   facecolor="#3f7a3f", edgecolor="none"))

    grid(4, 12, 1, "prior grid (t-1)")
    grid(70, 12, 2, "posterior grid (t)")

    def op(x, label):
        ax.add_patch(plt.Rectangle((x, 18), 13, 7, facecolor="#eef2f7", edgecolor="#9aa6b4", lw=1.1))
        ax.text(x + 6.5, 21.5, label, ha="center", va="center", fontsize=7.4)

    op(30, "1. warp\nSE(2) ego motion")
    op(46, "2. decay\nl <- gamma l")
    ax.annotate("", xy=(30, 21.5), xytext=(26, 21.5),
                arrowprops=dict(arrowstyle="-|>", color="#5a6068", lw=1.3))
    ax.annotate("", xy=(46, 21.5), xytext=(43, 21.5),
                arrowprops=dict(arrowstyle="-|>", color="#5a6068", lw=1.3))
    ax.annotate("", xy=(70, 21.5), xytext=(59, 21.5),
                arrowprops=dict(arrowstyle="-|>", color="#5a6068", lw=1.3))
    ax.text(64.5, 24.0, "3. fuse\nLiDAR hit/miss", ha="center", fontsize=7.4)
    ax.text(50, 6.5, "the wall moves one row closer as the ego drives forward; "
                     "evidence accumulates instead of resetting each frame",
            ha="center", fontsize=7.4, style="italic", color="#6e747e")
    ax.text(7, 30, "amber = occupied   navy = free   gray = unknown",
            fontsize=7.2, color="#6e747e")
    return _save(fig, "occupancy.png")


def diagram_lattice():
    import numpy as np
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.set_xlim(-13, 13)
    ax.set_ylim(-1.5, 17)
    ax.set_aspect("equal")
    ax.axis("off")

    # obstacle (blocked region) ahead and to the left
    ax.add_patch(plt.Rectangle((-9.5, 8.5), 5.5, 4.5, facecolor="#f4c9c2",
                               edgecolor="#b5382a", lw=1.2, zorder=2))
    ax.text(-6.75, 10.75, "blocked", ha="center", va="center", fontsize=8, color="#b5382a", zorder=3)

    def arc(curv, dist=15.0, n=40):
        xs, ys, x, y, yaw = [], [], 0.0, 0.0, np.pi / 2  # heading up
        step = dist / n
        for _ in range(n):
            if abs(curv) < 1e-9:
                x += step * np.cos(yaw); y += step * np.sin(yaw)
            else:
                nyaw = yaw + curv * step
                x += (np.sin(nyaw) - np.sin(yaw)) / curv
                y += (-np.cos(nyaw) + np.cos(yaw)) / curv
                yaw = nyaw
            xs.append(x); ys.append(y)
        return np.array(xs), np.array(ys)

    def hits_obstacle(xs, ys):
        return np.any((xs >= -9.5) & (xs <= -4.0) & (ys >= 8.5) & (ys <= 13.0))

    curvs = [-0.13, -0.09, -0.05, 0.0, 0.05, 0.09, 0.13]
    selected = 0.05
    for cu in curvs:
        xs, ys = arc(cu)
        if hits_obstacle(xs, ys):
            ax.plot(xs, ys, color="#d98c84", lw=1.4, zorder=2)
            ax.plot(xs[-1], ys[-1], "x", color="#b5382a", ms=5, zorder=3)
        elif cu == selected:
            ax.plot(xs, ys, color="#2e8b3d", lw=2.6, zorder=4)
            ax.plot(xs[-1], ys[-1], "o", color="#2e8b3d", ms=5, zorder=4)
        else:
            ax.plot(xs, ys, color="#9aa0a8", lw=1.4, zorder=2)

    ax.add_patch(plt.Rectangle((-0.7, -1.2), 1.4, 2.4, facecolor="#3f7a3f",
                               edgecolor="#26492a", zorder=5))
    ax.add_patch(plt.Circle((0, 0), 1.25, facecolor="none", edgecolor="#3f7a3f",
                            lw=1.0, ls="--", zorder=4))
    ax.text(8.5, 1.5, "green = selected\n(lowest cost)", fontsize=7.6, color="#2e8b3d")
    ax.text(8.5, 13.5, "red = rejected\n(collision)", fontsize=7.6, color="#b5382a")
    ax.text(0, -2.0, "ego: dashed disc = 1.25 m footprint radius",
            ha="center", va="top", fontsize=7.2, color="#6e747e")
    return _save(fig, "lattice.png")


# ---- pdf builder ------------------------------------------------------------
class Doc(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_margins(18, 18, 18)
        self.set_auto_page_break(True, margin=20)
        self.add_font("A", "", str(FONTS / "arial.ttf"))
        self.add_font("A", "B", str(FONTS / "arialbd.ttf"))
        self.add_font("A", "I", str(FONTS / "ariali.ttf"))
        self.add_font("M", "", str(FONTS / "consola.ttf"))
        self.add_font("M", "B", str(FONTS / "consolab.ttf"))
        self._title_mode = False
        self._eqno = 0

    def footer(self):
        if self._title_mode:
            return
        self.set_y(-14)
        self.set_font("A", "", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 6, f"360 Vision  ·  Technical Whitepaper  ·  {self.page_no()}", align="C")

    # primitives --------------------------------------------------------------
    def ensure(self, h):
        if self.get_y() + h > self.h - self.b_margin:
            self.add_page()

    def h1(self, text, num=None):
        self.add_page()
        self.set_font("A", "B", 9)
        self.set_text_color(*ACCENT)
        if num is not None:
            self.cell(0, 6, f"SECTION {num}", new_x="LMARGIN", new_y="NEXT")
        self.set_font("A", "B", 21)
        self.set_text_color(*INK)
        self.multi_cell(0, 9, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1.5)
        self.set_draw_color(*ACCENT)
        self.set_line_width(0.6)
        y = self.get_y()
        self.line(18, y, 18 + CONTENT_W, y)
        self.set_line_width(0.2)
        self.ln(4)

    def h2(self, text):
        self.ensure(16)
        self.ln(2)
        self.set_font("A", "B", 13.5)
        self.set_text_color(*INK)
        self.multi_cell(0, 6.5, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1.5)

    def h3(self, text):
        self.ensure(12)
        self.ln(1)
        self.set_font("A", "B", 11)
        self.set_text_color(*ACCENT2)
        self.multi_cell(0, 5.5, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(0.5)

    def para(self, text):
        self.set_font("A", "", 10.3)
        self.set_text_color(*INK)
        self.multi_cell(0, 5.3, text, markdown=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(1.6)

    def bullet(self, text):
        self.set_font("A", "", 10.3)
        self.set_text_color(*INK)
        x = self.get_x()
        self.set_x(x + 3)
        self.set_font("A", "B", 10.3)
        self.cell(4, 5.1, "·")
        self.set_font("A", "", 10.3)
        self.multi_cell(0, 5.1, text, markdown=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(0.6)

    def caption(self, text):
        self.set_font("A", "I", 8.8)
        self.set_text_color(*MUTED)
        self.multi_cell(0, 4.3, text, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(1.5)

    def code(self, text, caption=None):
        text = text.strip("\n")
        lines: list[str] = []
        for raw in text.split("\n"):
            raw = raw.replace("\t", "    ")
            while len(raw) > 96:
                lines.append(raw[:96])
                raw = "    " + raw[96:]
            lines.append(raw)
        lh = 4.05
        pad = 2.4
        box_h = len(lines) * lh + 2 * pad
        self.ensure(box_h + 4)
        x, y = self.get_x(), self.get_y()
        self.set_fill_color(*CODE_BG)
        self.set_draw_color(*RULE)
        self.rect(x, y, CONTENT_W, box_h, style="DF")
        self.set_xy(x + pad, y + pad)
        self.set_font("M", "", 7.7)
        self.set_text_color(*CODE_INK)
        for ln in lines:
            self.set_x(x + pad)
            self.cell(CONTENT_W - 2 * pad, lh, ln, new_x="LMARGIN", new_y="NEXT")
        self.set_y(y + box_h + 1.5)
        if caption:
            self.caption(caption)
        else:
            self.ln(1.5)

    def equation(self, latex, fontsize=12.5, number=True):
        dpi = 300
        png = eq_png(latex, fontsize=fontsize, dpi=dpi)
        w_px, h_px = Image.open(png).size
        w_mm = w_px / dpi * 25.4
        h_mm = h_px / dpi * 25.4
        max_w = 124.0  # research-paper display width, not full content width
        if w_mm > max_w:
            h_mm *= max_w / w_mm
            w_mm = max_w
        self.ensure(h_mm + 7)
        # centered display block with a little breathing room above/below
        self.ln(2.0)
        y0 = self.get_y()
        self.image(str(png), x=18 + (CONTENT_W - w_mm) / 2, y=y0, w=w_mm, h=h_mm)
        if number:
            self._eqno += 1
            self.set_font("A", "", 9.5)
            self.set_text_color(*MUTED)
            self.set_xy(18, y0 + h_mm / 2 - 2.2)
            self.cell(CONTENT_W, 4.4, f"({self._eqno})", align="R")
        self.set_y(y0 + h_mm)
        self.ln(3.0)
        self.set_text_color(*INK)

    def image_block(self, png, width_frac=0.92, caption=None):
        w_px, h_px = Image.open(png).size
        w_mm = CONTENT_W * width_frac
        h_mm = h_px / w_px * w_mm
        self.ensure(h_mm + 8)
        self.ln(1)
        y0 = self.get_y()
        # pass y explicitly and set the cursor absolutely: fpdf2's image() already
        # advances y by the height, so adding it again double-counts and strands a
        # full image-height of blank space below the figure.
        self.image(str(png), x=18 + (CONTENT_W - w_mm) / 2, y=y0, w=w_mm, h=h_mm)
        self.set_y(y0 + h_mm + 1.5)
        if caption:
            self.caption(caption)


def build():
    d = Doc()

    # ---- title page ---------------------------------------------------------
    d._title_mode = True
    d.add_page()
    d.set_y(46)
    d.set_font("A", "B", 9)
    d.set_text_color(*ACCENT)
    d.cell(0, 6, "TECHNICAL WHITEPAPER", new_x="LMARGIN", new_y="NEXT", align="C")
    d.ln(4)
    d.set_font("A", "B", 30)
    d.set_text_color(*INK)
    d.multi_cell(0, 13, "A LiDAR-Camera Bird's-Eye-View World Model on nuScenes", align="C",
                 new_x="LMARGIN", new_y="NEXT")
    d.ln(3)
    d.set_font("A", "", 12.5)
    d.set_text_color(*MUTED)
    d.multi_cell(0, 6,
                 "The 360-degree surround-vision arm of the vision-fsd project: geometry, "
                 "math, and implementation from raw sensors to a stateful BEV world model and "
                 "a classical local planner.",
                 align="C", new_x="LMARGIN", new_y="NEXT")
    d.ln(10)
    d.image_block(diagram_pipeline(), width_frac=1.0)
    d.set_y(-40)
    d.set_font("A", "", 9.5)
    d.set_text_color(*MUTED)
    d.multi_cell(0, 5,
                 "Scope: the nuScenes 360 pipeline only (fsd/). The monocular dashcam stack is "
                 "deliberately excluded. Every section maps to source in fsd/ and is written to "
                 "be reproducible from the code.",
                 align="C", new_x="LMARGIN", new_y="NEXT")
    d._title_mode = False

    # ===================================================================== 1
    d.h1("System Overview", num=1)
    d.para(
        "The 360 pipeline turns one nuScenes keyframe - six surround cameras, one 32-beam "
        "LiDAR sweep, and the vehicle's ego pose - into a **bird's-eye-view (BEV) world "
        "model**: a top-down, metric, ego-centred description of the scene that a planner "
        "can reason over. The defining design move is the shift from a **frame-centric** "
        "system (process a frame, draw it, discard it) to a **stateful** one that warps and "
        "accumulates evidence across time.")
    d.para(
        "Everything is built on three coordinate frames and the rigid transforms between "
        "them, so Section 2 establishes that machinery first. The rest of the document then "
        "follows the data: LiDAR geometry and BEV rasterization (3), the 2.5D height tensor "
        "(4), temporal occupancy fusion (5), 3D object detection by three independent routes "
        "(6), the camera-only Lift-Splat-Shoot BEV (7), the unified world model (8), and the "
        "classical local planner (9).")
    d.h3("Module map")
    d.para("Each capability is one module under `fsd/`. The visualizer (`fsd/visualize.py`) "
           "selects a *view* and drives the per-frame render loop.")
    d.code(
        "fsd/data.py              read-only nuScenes loader (6 cams + LiDAR, lazy)\n"
        "fsd/lidar_projection.py  quaternion/rigid transforms, LiDAR->camera, pinhole\n"
        "fsd/bev.py               ego-frame LiDAR BEV rasterizer\n"
        "fsd/bev_tensor.py        2.5D per-cell height-channel tensor\n"
        "fsd/occupancy.py         temporal log-odds occupancy fusion\n"
        "fsd/object_detection.py  GT 3D boxes + prediction-JSON adapter + BEV drawing\n"
        "fsd/fusion_detect.py     camera(YOLO)+LiDAR frustum-fusion detector\n"
        "fsd/centerpoint_export.py  pretrained CenterPoint (isolated mmdet3d env)\n"
        "fsd/lss.py               Lift-Splat-Shoot camera-only BEV segmentation\n"
        "fsd/world_model.py       unified BevWorldModel\n"
        "fsd/motion_planning/     ego state, lattice sampler, validator, costs, planner")

    # ===================================================================== 2
    d.h1("Data and Coordinate Frames", num=2)
    d.para(
        "nuScenes is ~850 independent 20-second clips (*scenes*). Each scene is a chain of "
        "*samples* (keyframes) at 2 Hz, plus higher-rate intermediate *sweeps*. A sample "
        "links to one `sample_data` record per sensor channel, and each record carries a "
        "`calibrated_sensor` (sensor->ego extrinsic + camera intrinsic) and an `ego_pose` "
        "(ego->global pose at that timestamp).")

    d.h2("2.1  Read-only streaming loader")
    d.para(
        "The dataset is treated as external and read-only: nothing is copied into the repo, "
        "and there is **no nuscenes-devkit runtime dependency** in `fsd/`. Metadata is read "
        "straight from the JSON tables. The large tables (`sample_data.json`, `ego_pose.json`, "
        "`sample_annotation.json`) are hundreds of MB, so `data.py` streams them object by "
        "object instead of loading the whole array, stopping as soon as the requested tokens "
        "are found.")
    d.code(
        "def _iter_json_objects(path):\n"
        "    with path.open('r', encoding='utf-8') as handle:\n"
        "        collecting, depth, lines = False, 0, []\n"
        "        for raw_line in handle:\n"
        "            stripped = raw_line.strip()\n"
        "            if not collecting:\n"
        "                if stripped.startswith('{'):\n"
        "                    collecting = True; lines = [raw_line]\n"
        "                    depth = raw_line.count('{') - raw_line.count('}')\n"
        "                continue\n"
        "            lines.append(raw_line)\n"
        "            depth += raw_line.count('{') - raw_line.count('}')\n"
        "            if depth == 0:\n"
        "                text = ''.join(lines).strip().rstrip(',')\n"
        "                yield json.loads(text); collecting = False; lines = []",
        caption="fsd/data.py - brace-counting streaming parser; tokens are cached on first hit.")
    d.para(
        "Frames are exposed as immutable dataclasses: `CameraFrame` (image path + intrinsic + "
        "extrinsic + ego pose), `LidarFrame` (point-cloud path + extrinsic + ego pose), and "
        "`SurroundFrame` bundling the six cameras for one sample. Image and point-cloud files "
        "are loaded lazily, only when a view actually needs them.")

    d.h2("2.2  The three frames")
    d.bullet("**Sensor frame** - raw measurements (LiDAR points, camera rays) in the sensor's "
             "own axes.")
    d.bullet("**Ego frame** - the vehicle body frame at one timestamp: x forward, y left, "
             "z up. The world model lives here.")
    d.bullet("**Global frame** - a fixed map frame per log. Ego motion across time is "
             "expressed here, which is what makes temporal fusion possible.")
    d.para("Every transform is rigid: a rotation R and a translation t. nuScenes stores "
           "rotations as unit quaternions in **w, x, y, z** order. The conversion to a 3x3 "
           "matrix (after normalising q):")
    d.code(
        "w, x, y, z = q / |q|\n"
        "R = | 1-2(y2+z2)   2(xy-zw)    2(xz+yw) |\n"
        "    | 2(xy+zw)     1-2(x2+z2)  2(yz-xw) |\n"
        "    | 2(xz-yw)     2(yz+xw)    1-2(x2+y2) |",
        caption="fsd/lidar_projection.py - quaternion_to_rotation_matrix (x2 == x squared).")
    d.para("A source->target transform and its exact inverse are the two workhorses of the "
           "whole codebase (points are stored as row vectors, hence the transpose):")
    d.equation(r"\mathbf{p}_{\mathrm{tgt}} = \mathbf{p}_{\mathrm{src}}\,R^{\top} + \mathbf{t}"
               r"\qquad\Longleftrightarrow\qquad"
               r"\mathbf{p}_{\mathrm{src}} = (\mathbf{p}_{\mathrm{tgt}} - \mathbf{t})\,R")
    d.para("These two functions - `transform_points` and `inverse_transform_points` - compose "
           "into every sensor->ego->global->sensor chain that follows.")

    d.image_block(diagram_frames(), width_frac=1.0,
                  caption="Figure 1. The three coordinate frames and the rigid transforms between "
                          "them. LiDAR points flow sensor -> ego -> global; cameras invert the chain.")

    # ===================================================================== 3
    d.h1("LiDAR Geometry: Projection and BEV", num=3)
    d.para(
        "A nuScenes LIDAR_TOP sweep is a binary blob of 5 floats per point "
        "(x, y, z, intensity, ring). Only XYZ is used for geometry; the cloud is reshaped to "
        "Nx5 and sliced to Nx3.")

    d.h2("3.1  LiDAR into the cameras")
    d.para("To colour a camera image by LiDAR depth, sensor-frame points are walked through "
           "the full chain into the camera frame: LiDAR-sensor -> ego(LiDAR time) -> global "
           "-> ego(camera time) -> camera-sensor. The first two hops are forward transforms; "
           "the last two are inverse transforms.")
    d.code(
        "points = transform_points(lidar_pts, lidar.cs.rot, lidar.cs.trans)      # -> ego\n"
        "points = transform_points(points,    lidar.ego.rot, lidar.ego.trans)    # -> global\n"
        "points = inverse_transform_points(points, cam.ego.rot, cam.ego.trans)   # -> ego(cam)\n"
        "points = inverse_transform_points(points, cam.cs.rot,  cam.cs.trans)    # -> camera",
        caption="fsd/lidar_projection.py - lidar_points_to_camera.")
    d.para("Camera-frame points are then projected to pixels with the pinhole intrinsic K. "
           "Points behind the image plane (z <= min_depth) are dropped first:")
    d.equation(r"\mathbf{u} = \frac{1}{z}\,K\,\mathbf{p}_{\mathrm{cam}},"
               r"\qquad u = f_x\frac{x}{z} + c_x,\quad v = f_y\frac{y}{z} + c_y")
    d.para("Depth is mapped through a TURBO colormap (near = warm) so the projected points "
           "read as a depth image overlaid on each of the six cameras.")

    d.h2("3.2  Ego-frame BEV rasterization")
    d.para("The BEV is a top-down grid over a metric window - default x,y in [-50, 50] m at "
           "0.25 m/cell, giving a 400x400 grid. Sensor points are first moved into the ego "
           "frame, then each surviving point's metric (x, y) maps to an integer (row, col). "
           "Forward (+x) is up, left (+y) is left, so:")
    d.equation(r"\mathrm{row} = \left\lfloor\frac{x_{\max}-x}{\Delta}\right\rfloor,"
               r"\qquad \mathrm{col} = \left\lfloor\frac{y_{\max}-y}{\Delta}\right\rfloor")
    d.image_block(diagram_bev_grid(), width_frac=0.62,
                  caption="Figure 2. Ego-frame BEV rasterization. Each metric (x, y) maps to an "
                          "integer (row, col); forward is up, left is left. The same map is reused "
                          "by every BEV layer so they overlay cleanly.")
    d.para("with grid resolution **Delta** (metres per cell). Points outside the window or "
           "outside a z-band are masked out; the rest are coloured by height (TURBO) and "
           "written to the canvas. This same metric->pixel map, with the same origin "
           "convention, is reused by the height tensor, the occupancy grid, and box drawing - "
           "consistency here is what lets the layers overlay cleanly.")

    # ===================================================================== 4
    d.h1("2.5D BEV Height-Channel Tensor", num=4)
    d.para(
        "A plain BEV only records 'cell has points or not'. The 2.5D tensor keeps the "
        "vertical structure that a single occupancy bit throws away. Every channel is derived "
        "from the LiDAR sweep alone - no labels, no map - so it is deployable on a real "
        "vehicle. It is the data structure downstream modules read instead of re-rasterizing "
        "raw points.")
    d.h3("Per-cell channels")
    d.bullet("**density** - point count in the cell (surface support / confidence).")
    d.bullet("**max_height / min_height / mean_height** - vertical extremes and average of "
             "returns in the cell.")
    d.bullet("**height_range** = max_height - min_height - the discriminative one.")
    d.para("The key insight: **height_range** separates drivable surface from obstacles with "
           "pure geometry. Flat road produces returns at nearly one height, so its range is "
           "~0; a car or wall spans a large vertical extent, so its range is large. No "
           "learning required.")
    d.equation(r"h_{\mathrm{rng}}(c) = \max_{i\in c} z_i - \min_{i\in c} z_i,"
               r"\qquad \bar{z}(c) = \frac{1}{n_c}\sum_{i\in c} z_i")
    d.para("Rasterization is fully vectorized. Each point's (row, col) is flattened to a "
           "linear cell index, and unbuffered scatter-reductions accumulate the statistics in "
           "one pass over the cloud:")
    d.code(
        "flat = rows * w + cols\n"
        "count = np.zeros(h*w); np.add.at(count, flat, 1.0)          # density\n"
        "zsum  = np.zeros(h*w); np.add.at(zsum,  flat, z)            # for mean\n"
        "zmax  = np.full(h*w, -inf); np.maximum.at(zmax, flat, z)    # max height\n"
        "zmin  = np.full(h*w,  inf); np.minimum.at(zmin, flat, z)    # min height\n"
        "occupied = count > 0\n"
        "mean = where(occupied, zsum / maximum(count, 1), 0.0)\n"
        "height_range = where(occupied, zmax - zmin, 0.0)",
        caption="fsd/bev_tensor.py - compute_bev_height_channels. np.add.at / maximum.at are "
                "the unbuffered scatter ops that make repeated cell hits accumulate correctly.")
    d.para("`BevTensor.stack()` returns an (H, W, 5) float array in channel order. Verified "
           "synthetically: a 2.5 m wall column reads ~2.5 m height_range; flat ground reads "
           "~0.")

    # ===================================================================== 5
    d.h1("Temporal Occupancy Fusion", num=5)
    d.para(
        "This is the step that makes the system stateful. A rolling occupancy grid is kept in "
        "**log-odds** in the current ego frame. Each keyframe the previous grid is warped into "
        "the new ego frame, decayed toward 'unknown', and updated with fresh LiDAR evidence. "
        "The result is a continuously evolving map of free and occupied space rather than a "
        "single-frame snapshot.")

    d.h2("5.1  Log-odds occupancy")
    d.para("Each cell stores the log-odds of being occupied. Evidence is additive in log-odds "
           "(a Bayesian binary filter / inverse-sensor model), which is why the update is a "
           "simple add-and-clamp. The probability is recovered with the logistic sigmoid:")
    d.equation(r"l_{t} = \mathrm{clip}\left(\gamma\, l_{t-1}^{\,\mathrm{warp}} "
               r"+ \Delta l,\; -L,\; +L\right),"
               r"\qquad p = \sigma(l) = \frac{1}{1+e^{-l}}")
    d.para("A hit adds +logit_hit (0.85), a miss adds -logit_miss (0.4), the decay factor is "
           "gamma = 0.97, and log-odds are clamped to +/-5. Decay continually pulls unobserved "
           "cells back toward p = 0.5 (unknown), so stale evidence fades.")

    d.h2("5.2  Warping the prior into the current frame")
    d.para("Between two keyframes the ego moves. To keep the grid ego-centred, the previous "
           "grid must be resampled into the new frame. The relative planar motion is an SE(2) "
           "built from the two global ego poses - it maps a point expressed in the *current* "
           "ego frame back to the *previous* ego frame:")
    d.equation(r"R_{\mathrm{rel}} = R_{\mathrm{prev}}^{\top} R_{\mathrm{cur}},"
               r"\qquad \mathbf{t}_{\mathrm{rel}} = R_{\mathrm{prev}}^{\top}"
               r"(\mathbf{t}_{\mathrm{cur}} - \mathbf{t}_{\mathrm{prev}})")
    d.para("Because the grid is an image, this metric SE(2) is sandwiched between a "
           "pixel->metre map and a metre->pixel map to get a single 2x3 affine that "
           "`cv2.warpAffine` can apply:")
    d.equation(r"M = M_{\mathrm{m2p}}\;\cdot\;T_{\mathrm{rel}}^{\,\mathrm{SE(2)}}"
               r"\;\cdot\;M_{\mathrm{p2m}}")
    d.code(
        "se2 = [[R_rel, t_rel], [0, 0, 1]]\n"
        "p2m = [[0, -res,  x_max], [-res, 0, y_max], [0,0,1]]   # pixel -> metre\n"
        "m2p = [[0, -1/res, y_max/res], [-1/res, 0, x_max/res], [0,0,1]]  # metre -> pixel\n"
        "M = (m2p @ se2 @ p2m)[:2]\n"
        "logodds = cv2.warpAffine(logodds, M, (w, h),\n"
        "          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,  # M already maps cur->prev\n"
        "          borderValue=0.0)   # outside = unknown",
        caption="fsd/occupancy.py - WARP_INVERSE_MAP is essential: M already maps current-frame "
                "pixels to previous-frame pixels, and the flag stops OpenCV inverting it again.")

    d.h2("5.3  Evidence: a height split, not ray casting")
    d.para("Fresh evidence comes from a deliberately simple rule. Points are rasterized to "
           "cells; returns below ground_height (0.3 m in the ego frame) mark their cell "
           "**free**, taller returns mark it **occupied**. Occupied wins ties.")
    d.code(
        "ground   = inbounds & (z >= z_min) & (z <  ground_height)   # road surface -> free\n"
        "obstacle = inbounds & (z >= ground_height) & (z <= z_max)   # tall -> occupied\n"
        "free[rows[ground],  cols[ground]]  = True\n"
        "occ [rows[obstacle], cols[obstacle]] = True\n"
        "free[occ] = False\n"
        "logodds[free] += -logit_miss\n"
        "logodds[occ]  += +logit_hit",
        caption="fsd/occupancy.py - _evidence + step.")
    d.para("Why not ray-cast free space? An earlier polar 'nearest-hit' free-space polygon was "
           "dropped: ground returns are the nearest hit in nearly every direction, so casting "
           "free space up to the first hit both collapsed the free region and painted the road "
           "as occupied. Marking only observed cells, split by height, keeps the road out of "
           "the occupied set. Verified synthetically: 2 m forward motion moves a 20 m wall to "
           "18 m; a +90 deg yaw moves a point straight ahead to the right.")

    d.image_block(diagram_occupancy(), width_frac=1.0,
                  caption="Figure 3. One temporal occupancy update: warp the prior grid into the new "
                          "ego frame, decay it toward unknown, then fuse fresh LiDAR hit/miss evidence.")

    # ===================================================================== 6
    d.h1("3D Object Detection", num=6)
    d.para("Objects enter the world model as ego-frame `Box3D` records - center, size "
           "(w, l, h), yaw, and 2D/3D corners. Three independent routes produce them.")

    d.h2("6.1  Ground-truth boxes from annotations")
    d.para("nuScenes annotations are 3D boxes in the global frame (translation, size, "
           "rotation quaternion). Each is converted to the ego frame. The four bottom corners "
           "are built in the box's local frame, rotated into global, then inverse-transformed "
           "by the ego pose; the box's ego-frame yaw comes from the composed rotation:")
    d.equation(r"R_{\mathrm{box}}^{\mathrm{ego}} = R_{\mathrm{ego}}^{\top}\,"
               r"R_{\mathrm{box}}^{\mathrm{global}},\qquad"
               r"\psi_{\mathrm{ego}} = \mathrm{atan2}\left(R_{10},\,R_{00}\right)")
    d.para("Boxes are filtered by `num_lidar_pts >= min_lidar_points` so empty annotations are "
           "skipped, and the nuScenes category strings are folded into a small class set "
           "(car, truck, bus, trailer, motorcycle, bicycle, pedestrian, ...).")

    d.h2("6.2  CenterPoint - a real learned LiDAR detector")
    d.para("The canonical path is pretrained **CenterPoint** (mmdet3d). The OpenMMLab stack "
           "would not build on the main environment (Py3.12 / Torch 2.7 / CUDA 12.8, no "
           "prebuilt mmcv), so it lives in a separate pinned env (`.venv-mmdet3d`: Py3.11, "
           "Torch 2.1+cu121, mmcv 2.1, mmdet3d 1.4) and feeds boxes back through a "
           "prediction-JSON adapter.")
    d.h3("The multi-sweep fix")
    d.para("nuScenes CenterPoint is trained on **10 accumulated LiDAR sweeps** with the "
           "per-point 5th channel set to a time delta. Naively running inference on a single "
           ".pcd.bin feeds the model ~25k points instead of ~270k (the config's multi-sweep "
           "loader just pads by duplication), which visibly hurt recall and produced spurious "
           "boxes. The exporter aggregates the real 10 sweeps itself - transforming each older "
           "sweep into the current LiDAR frame and stamping its time delta - then strips the "
           "pipeline's `LoadPointsFromMultiSweeps` so it is not re-padded.")
    d.code(
        "ref_from_car  = T(ref_cs, inverse=True)      # current LiDAR sensor <- ego\n"
        "car_from_global = T(ref_pose, inverse=True)  # ego <- global\n"
        "for each older sweep sd:\n"
        "    global_from_car = T(cur_pose)            # global <- ego(sd)\n"
        "    car_from_current = T(cur_cs)             # ego(sd) <- sensor(sd)\n"
        "    pc.transform(ref_from_car @ car_from_global @ global_from_car @ car_from_current)\n"
        "    time = (ref_time - sd_time) * ones(N)    # per-point time channel\n"
        "    points = concat(points, [pc.xyz, intensity, time])",
        caption="fsd/centerpoint_export.py - _aggregate_sweeps. Effect on scene 0: ~25k -> "
                "~270k pts/frame, fewer false positives, predictions sit tightly on GT.")
    d.para("CenterPoint's boxes come out in the LiDAR sensor frame; the exporter converts each "
           "center and yaw to the ego frame with the LiDAR extrinsic and writes "
           "{samples: {sample_token: [{center_ego, yaw, size, detection_name, "
           "detection_score}]}} for the visualizer.")

    d.h2("6.3  Frustum-fusion detector - camera detects, LiDAR ranges")
    d.para("A second, label-free detector runs entirely on the main environment. Cameras "
           "answer *what and where in the image* (YOLO 2D boxes + COCO class); LiDAR answers "
           "*how far and how big*. It is 'frustum' detection without a learned 3D head: the 2D "
           "box defines a viewing frustum, and the LiDAR points falling inside it localise the "
           "object in 3D.")
    d.para("Per camera, per 2D detection:")
    d.bullet("Project the LiDAR sweep into the camera; keep the points whose pixels land "
             "inside the 2D box (the frustum).")
    d.bullet("Drop ground (z <= 0.3 m). A 2D box also catches background seen around the "
             "object, so keep only the **nearest cluster**: points within a depth band of a "
             "robust near range (the 15th percentile of range).")
    d.equation(r"r_i = \sqrt{x_i^2 + y_i^2},\qquad"
               r"\mathrm{keep}\;\; r_i \leq \mathrm{P}_{15}(r) + \mathrm{band}")
    d.bullet("Position = cluster XY mean; the box is rested on the ground plane at z = h/2 "
             "from a class size prior.")
    d.bullet("Heading from the cluster's dominant horizontal axis via PCA - the eigenvector of "
             "the 2x2 scatter matrix with the largest eigenvalue (radial fallback for tiny "
             "clusters):")
    d.equation(r"C = X^{\top}X,\quad C\,\mathbf{e}_k = \lambda_k \mathbf{e}_k,\quad"
               r"\psi = \mathrm{atan2}\left(\mathbf{e}_{\max,y},\,\mathbf{e}_{\max,x}\right)")
    d.para("Finally, objects seen by two overlapping cameras are de-duplicated greedily by "
           "class and centre distance (highest score wins). Honest limits: positions are as "
           "good as the LiDAR cluster (solid), but class is YOLO/COCO only, size is a prior "
           "not a measurement, and heading from a partial-view cluster is rough - good enough "
           "to feed tracking and velocity next.")

    # ===================================================================== 7
    d.h1("Camera-only BEV: Lift-Splat-Shoot", num=7)
    d.para("Lift-Splat-Shoot (LSS, ECCV 2020) is the first learned BEV head in the project: a "
           "camera-only model that predicts a top-down vehicle-segmentation map from the six "
           "images alone. The model code is a 1:1 port of the NVIDIA repo; the inference "
           "wrapper consumes our `SurroundFrame` directly instead of the devkit dataloader.")
    d.h3("Lift")
    d.para("Each camera image is encoded (EfficientNet-B0 trunk) into, per downsampled pixel, "
           "a categorical **depth distribution** over D discrete depth bins (softmax) and a "
           "C-dim context feature. The outer product of the two 'lifts' the 2D feature into a "
           "frustum of 3D features - feature times depth-probability at every (pixel, depth):")
    d.equation(r"\alpha \in \Delta^{D-1}\ (\text{softmax depth}),\qquad"
               r"\mathbf{c}_{d} = \alpha_d \,\mathbf{f}\ \in \mathbb{R}^{C}")
    d.code(
        "x = self.depthnet(features)                       # D + C channels\n"
        "depth = x[:, :D].softmax(dim=1)                   # depth distribution\n"
        "new_x = depth.unsqueeze(1) * x[:, D:D+C].unsqueeze(2)   # outer product -> (C, D, ...)",
        caption="fsd/lss.py - CamEncode.get_depth_feat.")
    d.h3("Splat")
    d.para("Each frustum point is placed in 3D using the camera intrinsic and extrinsic "
           "(`get_geometry`), then dropped into a fixed BEV voxel grid. Many frustum points "
           "fall in the same voxel, so features are **sum-pooled** per voxel. The efficiency "
           "trick: sort points by voxel rank and use a cumulative-sum so each voxel's total is "
           "one subtraction of cumsum endpoints - the autograd-friendly `QuickCumsum`.")
    d.code(
        "ranks = geom[:,0]*(nx1*nx2*B) + geom[:,1]*(nx2*B) + geom[:,2]*B + geom[:,3]\n"
        "x, geom, ranks = sort_by(ranks)\n"
        "x, geom = QuickCumsum.apply(x, geom, ranks)   # sum features sharing a voxel\n"
        "final[geom...] = x                            # scatter into BEV grid",
        caption="fsd/lss.py - voxel_pooling.")
    d.h3("Shoot (not in the port)")
    d.para("The released LSS code ships only the lift+splat vehicle-seg head. The cost-map / "
           "template-trajectory 'shoot' stage from the paper is not in the open-source repo, "
           "so it is not in this port. The wrapper replicates the eval-time resize+center-crop "
           "(to 128x352) and the six per-camera intrinsics/extrinsics, runs the pretrained "
           "checkpoint, and returns a 200x200 @ 0.5 m/cell probability grid reoriented to "
           "match the LiDAR BEV. An HD-map backdrop (parsed directly from the nuScenes "
           "map-expansion vectors) can be drawn behind it.")

    # ===================================================================== 8
    d.h1("The Unified BEV World Model", num=8)
    d.para("`WorldModelBuilder.step()` composes the static and dynamic layers into one "
           "immutable `BevWorldModel`: temporal occupancy probability, the 2.5D height tensor, "
           "a derived collision grid, ego state, and object footprints (GT and/or prediction "
           "boxes). It is the handoff point between perception and planning.")
    d.h3("Collision grid = occupancy OR height")
    d.para("The single most planning-relevant derived layer fuses the occupancy probability "
           "with the height tensor. A cell is blocked if it is probably occupied **or** tall - "
           "the height channel catches obstacles the occupancy filter has not yet committed "
           "to, and vice versa:")
    d.equation(r"\mathrm{blocked}(c) = \left[p(c) \geq \tau_p\right]\;\vee\;"
               r"\left[h_{\mathrm{rng}}(c) \geq \tau_h\right]")
    d.code(
        "blocked = (occupancy_probability >= 0.62) | (height_range >= 0.45)",
        caption="fsd/motion_planning/occupancy.py - build_collision_grid (tau_p=0.62, tau_h=0.45 m).")
    d.para("The objects are still recomputed each frame - they are an *overlay*, not yet part "
           "of the model's memory (no track id, velocity, or history). Closing that gap "
           "(tracking -> velocity -> short-horizon prediction) is the next milestone.")

    # ===================================================================== 9
    d.h1("Classical Local Planner", num=9)
    d.para("The planner is deliberately classical and debuggable: estimate ego state, sample a "
           "lattice of timed trajectories, reject colliding ones against the collision grid, "
           "score the survivors, and pick the cheapest - with an emergency stop as fallback.")

    d.h2("9.1  Ego state from poses")
    d.para("Speed and yaw-rate are finite-differenced from consecutive ego poses (first frame "
           "falls back to zero):")
    d.equation(r"v = \frac{\sqrt{\Delta x^2 + \Delta y^2}}{\Delta t},\qquad"
               r"\dot{\psi} = \frac{\mathrm{wrap}(\psi_t - \psi_{t-1})}{\Delta t}")

    d.h2("9.2  Timed lattice of trajectories")
    d.para("Candidates are the cross product of target speeds {0, 2.5, 5, 7.5} m/s and "
           "curvatures {+/-0.12, +/-0.06, +/-0.03, 0} 1/m, integrated over a 3 s horizon at "
           "0.25 s steps in the ego frame (start at origin, heading 0). Speed ramps linearly "
           "from the current speed to the target; each step advances along a "
           "**constant-curvature arc**:")
    d.equation(r"v(t) = v_0 + (v_{\mathrm{tgt}} - v_0)\,\frac{t}{T},\qquad"
               r"\Delta\psi = \kappa\, d,\quad d = \bar{v}\,\Delta t")
    d.equation(r"x' = x + \frac{\sin(\psi+\Delta\psi) - \sin\psi}{\kappa},\qquad"
               r"y' = y + \frac{\cos\psi - \cos(\psi+\Delta\psi)}{\kappa}")
    d.para("with the straight-line limit (x' = x + d cos psi, y' = y + d sin psi) used when "
           "|kappa| is ~0 to avoid dividing by zero.")

    d.image_block(diagram_lattice(), width_frac=0.6,
                  caption="Figure 4. The timed lattice. Constant-curvature arcs fan out from the ego; "
                          "arcs whose swept disc hits a blocked cell are rejected (red), and the "
                          "lowest-cost survivor is selected (green).")

    d.h2("9.3  Collision validation")
    d.para("Every trajectory point is checked as a **disc** of radius 1.25 m against the "
           "collision grid, and consecutive points are sub-sampled finely enough that the disc "
           "cannot tunnel through a thin obstacle between samples (spacing = min(cell/2, "
           "radius/2)). A footprint is blocked if any blocked cell lies within the radius, "
           "computed against the closest point of each candidate cell (a proper circle-vs-cell "
           "test, not just the centre). Leaving the grid counts as blocked.")
    d.code(
        "for step in 1..steps:                      # steps = ceil(seg_len / spacing)\n"
        "    p = lerp(start, end, step/steps)\n"
        "    if collision_grid.footprint_blocked(p.x, p.y, radius): return COLLISION\n"
        "# footprint_blocked: any blocked cell whose closest point is within radius",
        caption="fsd/motion_planning/validator.py - swept-disc collision check.")

    d.h2("9.4  Cost and selection")
    d.para("Valid trajectories are scored by a weighted sum that rewards progress and "
           "penalises lateral offset, curvature, and speed error (a lane-centering term "
           "activates when a confident lane context is present, which is not wired yet):")
    d.equation(r"J = -w_p\,L \;+\; w_l\,|y_N| \;+\; w_\kappa\,|\kappa| \;+\; w_v\,"
               r"|v_{\mathrm{tgt}} - v_0|")
    d.para("with weights (progress 4.0, lateral 1.2, curvature 2.0, speed 0.4). The lowest-cost "
           "valid trajectory is selected. If **no** candidate is collision-free, the planner "
           "returns a decelerating emergency stop (straight-ahead, ramping speed to zero) and "
           "flags whether even that is collision-free. The `OfflinePlanningRuntime` ties it "
           "together per frame - occupancy + height -> collision grid, ego state, plan - and "
           "resets its temporal state at every scene boundary.")
    d.code(
        "occupancy   = mapper.step(lidar)                     # temporal log-odds\n"
        "height      = bev_tensor_from_lidar(lidar)           # 2.5D channels\n"
        "collision   = build_collision_grid(occupancy, height.height_range, ...)\n"
        "ego         = estimate_ego_state(pose, t, prev_pose, prev_t)\n"
        "world       = PlannerWorld(ego, collision.blocked, occupancy, ...)\n"
        "result      = planner.plan(world)                    # sample -> validate -> score",
        caption="fsd/motion_planning/runner.py - OfflinePlanningRuntime.step.")

    # ===================================================================== 10
    d.h1("Status and Next Steps", num=10)
    d.para("What exists today is a complete static world model plus a working object overlay "
           "and a first classical planner:")
    d.bullet("**Stateful** temporal occupancy, 2.5D height tensor, and a fused collision grid.")
    d.bullet("Three object-detection routes: GT boxes, pretrained CenterPoint, and the "
             "label-free camera+LiDAR frustum-fusion detector.")
    d.bullet("Camera-only LSS BEV with an HD-map backdrop, and the unified `BevWorldModel`.")
    d.bullet("A lattice local planner over the collision grid with swept-disc validation, "
             "cost ranking, and an emergency-stop fallback.")
    d.para("The clear gap is **time-aware dynamic objects**. The model sees the present: it "
           "can say 'there is a car here', but not 'that car is moving at 6 m/s and will cross "
           "the ego path in 2 s'. The next milestone gives objects identity and motion - "
           "finite-difference velocity (GT instance tokens first as an oracle, then "
           "association over CenterPoint predictions), constant-velocity short-horizon "
           "prediction, and object-aware collision risk - after which the planner can reason "
           "about other agents, not just static free space.")

    d.output(str(OUT))
    print(OUT)


if __name__ == "__main__":
    build()
