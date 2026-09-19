"""
Small inline figure for a slide: the LOS-to-EW/Up decomposition geometry
(project.md Section 6.4), sized to drop into a slide at roughly 1/8 page,
portrait. Not a standalone captioned figure like fig01/fig03 - no title,
credit or disclaimer baked in; transparent background so it sits cleanly on
top of existing slide content.

Geometry is real, not illustrative: the two incidence angles and look-vector
components are the actual EGMS values for the Munich area used throughout
presentation/ - ascending track 044 (IW1, incidence 32.68 deg, los_east
-0.523, los_up 0.846 - the same values verified earlier for fig01/fig04) and
descending track 095 (IW3, incidence ~45.0 deg, los_east 0.698, los_up 0.707
- pulled fresh from EGMS_L2b_Descending.zip for this figure). Drawn as the
East-Up projection of each look vector (the small North component, ~-0.10 for
both, is dropped for the 2D cross-section - standard simplification, and
exactly the reason North isn't recoverable from this pair at all, §10.2).

Output: presentation/figs/fig06_decomposition_geometry.png (transparent, ~3x4.2in)
Run:    module load uv && uv run python fig06_decomposition_geometry.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, FancyArrowPatch

FIGS_DIR = Path(__file__).resolve().parent / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = FIGS_DIR / "fig06_decomposition_geometry.png"

# real EGMS geometry, Munich area (see docstring)
ASC_INC_DEG, ASC_E, ASC_U = 32.68, -0.523, 0.846
DESC_INC_DEG, DESC_E, DESC_U = 44.98, 0.698, 0.707

INK = "#111111"
ASC_COLOR = "#1c4f9c"
DESC_COLOR = "#8c1c3f"
AXIS_COLOR = "#666666"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
})


def unit(e, u):
    v = np.array([e, u])
    return v / np.linalg.norm(v)


def main():
    # Figure size and axes position are computed, not guessed: the content
    # (a wide V of two LOS rays + a horizontal equation line) is inherently
    # WIDER than tall (~1.2:1) - forcing it into a portrait canvas always
    # leaves some margin, no matter how the axes box is sized. What matters
    # is that margin being symmetric (centred) rather than dumped entirely
    # below the content, which is what a naive aspect="equal" box did here
    # originally.
    fig_w = 3.0
    fig_h = 3.4  # ~1/8 slide area (10.2 sq in of a 13.333x7.5in slide), portrait
    data_xlim = (-1.15, 1.15)
    data_ylim = (-0.62, 1.32)
    data_w = data_xlim[1] - data_xlim[0]
    data_h = data_ylim[1] - data_ylim[0]

    side_margin_frac = 0.05
    box_w_frac = 1 - 2 * side_margin_frac
    box_w_in = box_w_frac * fig_w
    box_h_in = box_w_in * (data_h / data_w)   # exact match to data aspect - no internal letterboxing
    box_h_frac = box_h_in / fig_h
    bottom_frac = (1 - box_h_frac) / 2         # centre the box vertically

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
    ax = fig.add_axes([side_margin_frac, bottom_frac, box_w_frac, box_h_frac])
    ax.set_xlim(*data_xlim)
    ax.set_ylim(*data_ylim)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # ground surface + point
    ax.plot([-1.3, 1.3], [0, 0], color=INK, linewidth=1.6, zorder=2)
    ax.hatch = None
    for x0 in np.linspace(-1.28, 1.28, 22):
        ax.plot([x0, x0 - 0.06], [0, -0.08], color=INK, linewidth=0.6, alpha=0.6, zorder=1)
    ax.scatter([0], [0], s=32, color=INK, zorder=5)

    # local East / Up reference axes (dashed, light)
    ax.annotate("", xy=(0.95, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=AXIS_COLOR, lw=1.0, ls=(0, (3, 2))))
    ax.text(1.0, -0.03, "E", color=AXIS_COLOR, fontsize=9, ha="left", va="top")
    ax.annotate("", xy=(0, 1.18), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=AXIS_COLOR, lw=1.0, ls=(0, (3, 2))))
    ax.text(0.04, 1.20, "Up", color=AXIS_COLOR, fontsize=9, ha="left", va="bottom")

    # LOS rays (ground -> satellite direction), real geometry
    ray_len = 1.18
    asc_end = unit(ASC_E, ASC_U) * ray_len
    desc_end = unit(DESC_E, DESC_U) * ray_len

    for end, color, label, inc in [
        (asc_end, ASC_COLOR, "ASC", ASC_INC_DEG),
        (desc_end, DESC_COLOR, "DESC", DESC_INC_DEG),
    ]:
        ax.add_patch(FancyArrowPatch((0, 0), tuple(end), arrowstyle="-|>",
                                      mutation_scale=12, color=color, linewidth=2.0, zorder=4))
        lx, ly = end * 1.13
        ha = "right" if end[0] < 0 else "left"
        ax.text(lx, ly, f"{label}\n{inc:.0f}°", color=color, fontsize=8.3,
                ha=ha, va="bottom", fontweight="bold", linespacing=1.15)

    # small incidence-angle arcs between the Up axis and each ray. Arc's
    # theta1/theta2 use the standard math convention (CCW from +x axis), so
    # each ray's angle is atan2(y, x) directly - NOT atan2(x, y), which was
    # the earlier bug that made both arcs compute nearly the same wedge.
    up_angle = 90.0
    for end, color in [(asc_end, ASC_COLOR), (desc_end, DESC_COLOR)]:
        ray_angle = np.degrees(np.arctan2(end[1], end[0]))
        t1, t2 = sorted([up_angle, ray_angle])
        arc = Arc((0, 0), 0.56, 0.56, angle=0, theta1=t1, theta2=t2,
                  color=color, lw=1.1, zorder=3)
        ax.add_patch(arc)

    # true 3D displacement vector being decomposed (illustrative direction)
    ax.add_patch(FancyArrowPatch((0, 0), (0.34, 0.62), arrowstyle="-|>",
                                  mutation_scale=10, color=INK, linewidth=1.6,
                                  linestyle=(0, (1, 1)), zorder=6))
    ax.text(0.37, 0.64, "true\nmotion", color=INK, fontsize=7.2, ha="left", va="bottom",
            linespacing=1.1, style="italic")

    # compact equation panel - placed just below the ground line, in the same
    # axes/data coordinates as the geometry, so there is no dead gap between
    # the two (the earlier two-separate-axes layout left a large blank strip)
    ax.text(0, -0.34, r"$\left(EW,\ Up\right) = G^{-1}\left(LOS_{asc},\ LOS_{desc}\right)$",
            fontsize=8.6, ha="center", va="center", color=INK, clip_on=False)
    ax.text(0, -0.50, "closed form — no learning", fontsize=6.8, ha="center", va="center",
            color="#555555", style="italic", clip_on=False)

    fig.savefig(OUT_PNG, transparent=True)
    print(f"Saved {OUT_PNG}  ({fig_w}x{fig_h} in @300dpi, transparent)")


if __name__ == "__main__":
    main()
