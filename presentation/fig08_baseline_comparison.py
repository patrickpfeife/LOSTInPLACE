"""
Small inline figure for a slide: baseline comparison (project.md Section 8.2
- regression kriging as the documented baseline; IDW added here as an even
simpler classical reference point, common in interpolation-method slides).
Same size/style family as fig06/fig07 - transparent background, no title
/credit/disclaimer, meant to drop directly into a slide.

The actual point being illustrated, not just "three methods exist": all
three panels show the SAME neighbourhood (identical points, identical
distances) - only the WEIGHTING RULE differs between rows. IDW and kriging
both weight purely by distance (different formulas, same input); the GNN's
attention weight (project.md Section 5.5) can depend on whatever the node/
edge features carry - land cover, coherence, direction - so its weights are
deliberately drawn NOT monotonic in distance here, unlike the other two rows.
This is a mechanism illustration, not a results claim - no accuracy numbers
are shown or implied, since none exist yet.

Output: presentation/figs/fig08_baseline_comparison.png (transparent, ~3x4.2in)
Run:    module load uv && uv run python fig08_baseline_comparison.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGS_DIR = Path(__file__).resolve().parent / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = FIGS_DIR / "fig08_baseline_comparison.png"

INK = "#111111"
IDW_COLOR = "#6b7280"
KRIGING_COLOR = "#3d6b8c"
GNN_COLOR = "#8c1c3f"
EDGE_COLOR = "#2b2b2b"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
})

# same neighbourhood for every row - fixed (angle_deg, distance) per point
NEIGHBOURS = [(20, 0.42), (100, 0.66), (200, 0.90), (260, 0.54), (330, 1.00)]
RADIUS = 0.5  # display radius of each mini cluster


def neighbour_xy(cx, cy, scale=RADIUS):
    pts = []
    for ang, d in NEIGHBOURS:
        r = ang * np.pi / 180
        pts.append((cx + scale * d * np.cos(r), cy + scale * d * np.sin(r)))
    return pts


def draw_cluster(ax, cx, cy, weights, color):
    pts = neighbour_xy(cx, cy)
    lw_max = 4.6
    w = np.array(weights, dtype=float)
    w_norm = w / w.max() * lw_max
    for (px, py), lw in zip(pts, w_norm):
        ax.plot([cx, px], [cy, py], color=EDGE_COLOR, linewidth=max(lw, 0.5),
                alpha=0.85, zorder=2, solid_capstyle="round")
    xs, ys = zip(*pts)
    ax.scatter(xs, ys, s=20, color="white", edgecolor=EDGE_COLOR, linewidth=1.0, zorder=4)
    ax.scatter([cx], [cy], s=48, color=color, edgecolor="white", linewidth=0.8, zorder=5)


def main():
    fig_w, fig_h = 3.0, 4.2
    data_xlim = (-1.15, 1.15)
    data_ylim = (-0.15, 3.35)
    data_w = data_xlim[1] - data_xlim[0]
    data_h = data_ylim[1] - data_ylim[0]

    side_margin_frac = 0.04
    box_w_frac = 1 - 2 * side_margin_frac
    box_w_in = box_w_frac * fig_w
    box_h_in = box_w_in * (data_h / data_w)
    box_h_frac = min(box_h_in / fig_h, 0.98)
    bottom_frac = (1 - box_h_frac) / 2

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
    ax = fig.add_axes([side_margin_frac, bottom_frac, box_w_frac, box_h_frac])
    ax.set_xlim(*data_xlim)
    ax.set_ylim(*data_ylim)
    ax.set_aspect("equal")
    ax.set_axis_off()

    dists = np.array([d for _, d in NEIGHBOURS])
    idw_w = 1.0 / dists
    krig_w = np.exp(-1.2 * dists)
    gnn_w = np.array([0.15, 0.55, 0.85, 0.35, 0.70])  # deliberately NOT distance-ordered

    rows = [
        (2.55, "IDW", r"$w \propto 1/d$", IDW_COLOR, idw_w,
         "distance only"),
        (1.55, "Kriging", r"$w$ from variogram($d$)", KRIGING_COLOR, krig_w,
         "distance only"),
        (0.55, "GNN (ours)", r"$w$ learned (attention)", GNN_COLOR, gnn_w,
         "+ land cover, coherence, ..."),
    ]

    cx = -0.55
    label_x = 0.05
    for cy, name, formula, color, weights, note in rows:
        draw_cluster(ax, cx, cy, weights, color)
        ax.text(label_x, cy + 0.20, name, color=color, fontsize=10.5,
                ha="left", va="center", fontweight="bold")
        ax.text(label_x, cy - 0.02, formula, color=INK, fontsize=7.6,
                ha="left", va="center")
        ax.text(label_x, cy - 0.24, note, color="#555555", fontsize=6.3,
                ha="left", va="center", style="italic")

    fig.savefig(OUT_PNG, transparent=True)
    print(f"Saved {OUT_PNG}  ({fig_w}x{fig_h} in @300dpi, transparent)")


if __name__ == "__main__":
    main()
