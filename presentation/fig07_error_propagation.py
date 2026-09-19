"""
Small inline figure for a slide: uncertainty/error propagation (project.md
Section 6), sized and styled to match fig06_decomposition_geometry.py -
transparent background, no title/credit/disclaimer, meant to drop directly
into a slide rather than stand alone.

Distinct from fig06 on purpose: fig06 is the geometric WHY (two look
vectors -> closed-form inversion); this one is the separate statistical
question of HOW MUCH to trust the answer, i.e. what happens to the
uncertainty itself as it moves through the pipeline:
  aleatoric (heteroscedastic head, Sec 6.1) + epistemic (deep ensemble,
  Sec 6.2) -> combined variance, propagated LINEARLY through basis
  reconstruction and the same G^-1 decomposition from fig06 (Sec 6.3/6.4,
  which is exactly why it CAN propagate in closed form rather than needing
  e.g. Monte Carlo) -> split-conformal calibration against held-out data
  (Sec 6.5) so the final interval is checked, not just asserted.

Deliberately uses its own small colour set (teal/amber/ink/green), distinct
from fig06's blue/bordeaux (orbit) and fig03's category palette, since this
figure encodes a different axis (uncertainty TYPE, not orbit or sampling
category) and reusing those hues here would imply a connection that isn't
there.

Output: presentation/figs/fig07_error_propagation.png (transparent, ~3x3.4in)
Run:    module load uv && uv run python fig07_error_propagation.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

FIGS_DIR = Path(__file__).resolve().parent / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = FIGS_DIR / "fig07_error_propagation.png"

INK = "#111111"
ALEATORIC_COLOR = "#1f8f8f"
EPISTEMIC_COLOR = "#c9861f"
TOTAL_COLOR = "#3a3a3a"
CALIB_COLOR = "#1f8a4c"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
})


def gaussian_curve(ax, x0, y0, width, height, color, alpha=0.55, zorder=3):
    xs = np.linspace(-1, 1, 120)
    ys = np.exp(-4.2 * xs**2)
    xs_d = x0 + xs * width
    ys_d = y0 + ys * height
    ax.fill_between(xs_d, y0, ys_d, color=color, alpha=alpha, zorder=zorder, linewidth=0)
    ax.plot(xs_d, ys_d, color=color, linewidth=1.4, zorder=zorder + 1)
    ax.plot([x0 - width, x0 + width], [y0, y0], color=color, linewidth=1.4, zorder=zorder + 1)


def arrow(ax, xy0, xy1, color=INK, lw=1.6):
    ax.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle="-|>", mutation_scale=9,
                                  color=color, linewidth=lw, zorder=5))


def main():
    fig_w = 3.0
    fig_h = 3.4
    data_xlim = (-1.3, 1.3)
    data_ylim = (-0.15, 3.35)
    data_w = data_xlim[1] - data_xlim[0]
    data_h = data_ylim[1] - data_ylim[0]

    side_margin_frac = 0.05
    box_w_frac = 1 - 2 * side_margin_frac
    box_w_in = box_w_frac * fig_w
    box_h_in = box_w_in * (data_h / data_w)
    box_h_frac = box_h_in / fig_h
    bottom_frac = (1 - box_h_frac) / 2

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
    ax = fig.add_axes([side_margin_frac, bottom_frac, box_w_frac, box_h_frac])
    ax.set_xlim(*data_xlim)
    ax.set_ylim(*data_ylim)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # row 1: aleatoric + epistemic
    y1 = 2.55
    gaussian_curve(ax, -0.55, y1, 0.40, 0.52, ALEATORIC_COLOR)
    gaussian_curve(ax, 0.55, y1, 0.40, 0.52, EPISTEMIC_COLOR)
    ax.text(-0.55, y1 - 0.10, r"aleatoric $\sigma_a$", color=ALEATORIC_COLOR, fontsize=6.6,
            ha="center", va="top", fontweight="bold")
    ax.text(0.55, y1 - 0.10, r"epistemic $\sigma_e$", color=EPISTEMIC_COLOR, fontsize=6.6,
            ha="center", va="top", fontweight="bold")

    # row 2: combined, propagated through Phi and G^-1. Arrows stay OUTSIDE
    # the centre column (x in [-0.35, 0.35]) their whole length, so they
    # never cross the equation text that lives in that column - the earlier
    # version converged the arrows toward the centre and ran them straight
    # through the equation.
    y2 = 1.40
    arrow(ax, (-0.50, 2.28), (-0.50, 1.90), color=ALEATORIC_COLOR)
    arrow(ax, (0.50, 2.28), (0.50, 1.90), color=EPISTEMIC_COLOR)
    gaussian_curve(ax, 0.0, y2, 0.60, 0.62, TOTAL_COLOR, alpha=0.32)
    ax.text(0, 2.10, r"$\sigma^2 = \sigma_a^2 + \sigma_e^2$", color=INK, fontsize=7.6,
            ha="center", va="center")
    ax.text(0, y2 - 0.16, "propagated through $\\Phi$, $G^{-1}$", color="#555555",
            fontsize=6.4, ha="center", va="top", style="italic")
    ax.text(0, y2 - 0.32, "(linear, closed form)", color="#555555",
            fontsize=6.4, ha="center", va="top", style="italic")

    # row 3: calibrated output
    y3 = 0.28
    arrow(ax, (0, y2 - 0.42), (0, y3 + 0.42), color=TOTAL_COLOR)
    ax.plot([-0.55, 0.55], [y3, y3], color=CALIB_COLOR, linewidth=2.4, zorder=4,
            solid_capstyle="round")
    ax.plot([-0.55, -0.55], [y3 - 0.06, y3 + 0.06], color=CALIB_COLOR, linewidth=2.4, zorder=4)
    ax.plot([0.55, 0.55], [y3 - 0.06, y3 + 0.06], color=CALIB_COLOR, linewidth=2.4, zorder=4)
    ax.scatter([0], [y3], s=26, color=CALIB_COLOR, zorder=5)
    ax.text(0.78, y3, "✓", color=CALIB_COLOR, fontsize=15, ha="left", va="center",
            fontweight="bold")
    ax.text(0, y3 - 0.24, "calibrated (conformal)", color=CALIB_COLOR, fontsize=6.8,
            ha="center", va="top", fontweight="bold")

    fig.savefig(OUT_PNG, transparent=True)
    print(f"Saved {OUT_PNG}  ({fig_w}x{fig_h} in @300dpi, transparent)")


if __name__ == "__main__":
    main()
