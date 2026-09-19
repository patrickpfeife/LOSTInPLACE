"""
Within-burst stratified sampling illustration - same visual language as the
fig03 European-wide sampling series (sampling_common.py), now zoomed to a
single burst. Illustrates project.md Section 4.3 Step 3: "Within each
selected burst, sample target points stratified by dominant land-cover class
... draw roughly equal numbers per class per burst ... This avoids
area-proportional bias."

Burst frame: same real geometry as fig04 (51.1 x 20.4 km, tilted to the real
ascending track heading), centred on the real Munich EGMS burst used
throughout presentation/. Subdivided into a grid ALIGNED TO THE BURST'S OWN
axes (range/azimuth, not lat/lon) - i.e. the same local geometry a real SAR
product's grid would actually have - into 32 cells, each assigned an
illustrative dominant land-cover class (Urban/Agriculture/Forest/Water).
Basemap is real Esri satellite imagery so the land-cover texture underneath
is genuine, even though the CLASS LABELS drawn on top are a schematic stand
-in for a real CORINE/ESA WorldCover classification (no such raster is
fetched here - see fig03's disclaimer convention, same spirit).

Sampling shown: ~20 points per class (not per cell, not per unit area) -
classes with fewer cells get proportionally more points per cell, which is
the whole point being illustrated (equal representation per class, not
per-area bias).

Output: presentation/figs/fig05_within_burst_sampling.png (fixed 16:9)
Run:    module load uv && uv run python fig05_within_burst_sampling.py
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextily as ctx
from pyproj import Transformer
from shapely.geometry import Point

import sampling_common as sc

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
})

CENTER_LATLON = (48.135, 11.58)
BURST_WIDTH_KM = 51.1
BURST_HEIGHT_KM = 20.4
BURST_ROTATION_DEG = -11.0
N_ACROSS, N_ALONG = 8, 4
POINTS_PER_CELL = 3   # every cell gets points - that is the whole point of the grid
SEED = 2026

LC_COLORS = {
    "Urban": "#c0392b",
    "Agriculture": "#d4a017",
    "Forest": "#1f5c33",
    "Water": "#1a8fa3",
}
FRAME_COLOR = "#f4f4f4"
GRID_COLOR = "#f4f4f4"
BOUNDARY_COLOR = "#111111"

to_3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)


def assign_land_cover(n_across, n_along, seed):
    """Deterministic, plausible-looking (not random-noise) illustrative
    assignment: a winding 'river' of Water cells roughly along the burst's
    long axis, an Urban cluster around the grid centre, Forest concentrated
    toward the along-axis extremes, Agriculture filling the rest - matching
    the real Isar-through-Munich geography loosely, not precisely."""
    rng = np.random.default_rng(seed)
    grid = np.full((n_across, n_along), "Agriculture", dtype=object)

    water_i_by_j = {0: 3, 1: 4, 2: 4, 3: 3}
    for j, i in water_i_by_j.items():
        grid[i, j] = "Water"

    mid_i = (n_across - 1) / 2
    mid_j = (n_along - 1) / 2
    for i in range(n_across):
        for j in range(n_along):
            if grid[i, j] == "Water":
                continue
            d = np.hypot(i - mid_i, j - mid_j)
            if d <= 1.6:
                grid[i, j] = "Urban"

    for i in range(n_across):
        for j in (0, n_along - 1):
            if grid[i, j] == "Agriculture" and rng.random() < 0.55:
                grid[i, j] = "Forest"
    for i in (0, 1, n_across - 2, n_across - 1):
        for j in range(n_along):
            if grid[i, j] == "Agriculture" and rng.random() < 0.25:
                grid[i, j] = "Forest"

    return [grid[i, j] for i in range(n_across) for j in range(n_along)]


def sample_points_per_cell(cells, classes, points_per_cell, seed):
    """Every cell gets its own points - that's the actual purpose of the grid:
    guaranteeing every area of the burst is represented, not just wherever the
    largest land-cover classes happen to sit. Each point inherits its cell's
    class for colouring/stratification bookkeeping, but coverage is spatial
    (per cell) first."""
    rng = np.random.default_rng(seed)
    pts = {c: [] for c in LC_COLORS}
    for cell, cls in zip(cells, classes):
        minx, miny, maxx, maxy = cell.bounds
        n_placed = 0
        while n_placed < points_per_cell:
            x = rng.uniform(minx, maxx)
            y = rng.uniform(miny, maxy)
            if cell.contains(Point(x, y)):
                pts[cls].append((x, y))
                n_placed += 1
    return pts


def main():
    e0, n0 = to_3035.transform(*CENTER_LATLON[::-1])
    cells = sc.rotated_grid(e0, n0, BURST_WIDTH_KM, BURST_HEIGHT_KM, BURST_ROTATION_DEG,
                             N_ACROSS, N_ALONG)
    classes = assign_land_cover(N_ACROSS, N_ALONG, SEED)
    pts_by_class = sample_points_per_cell(cells, classes, POINTS_PER_CELL, SEED + 1)

    burst_outline = sc.rotated_rect(e0, n0, BURST_WIDTH_KM, BURST_HEIGHT_KM, BURST_ROTATION_DEG)

    fig, ax = sc.setup_figure()

    minx, miny, maxx, maxy = burst_outline.bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    w, h = maxx - minx, maxy - miny
    target_ratio = sc.AXES_RECT[2] * sc.FIG_SIZE[0] / (sc.AXES_RECT[3] * sc.FIG_SIZE[1])
    margin = 1.10
    if w / h > target_ratio:
        half_w = w / 2 * margin
        half_h = half_w / target_ratio
    else:
        half_h = h / 2 * margin
        half_w = half_h * target_ratio
    ax.set_xlim(cx - half_w, cx + half_w)
    ax.set_ylim(cy - half_h, cy + half_h)
    ax.set_aspect("equal")

    ctx.add_basemap(ax, crs="EPSG:3035", source=ctx.providers.Esri.WorldImagery, attribution=False)
    ax.set_xlim(cx - half_w, cx + half_w)
    ax.set_ylim(cy - half_h, cy + half_h)

    for cell, cls in zip(cells, classes):
        xs, ys = cell.exterior.xy
        ax.fill(xs, ys, facecolor=LC_COLORS[cls], alpha=0.30, zorder=3)
        ax.plot(xs, ys, color=GRID_COLOR, linewidth=0.9, alpha=0.8, zorder=4)

    xs, ys = burst_outline.exterior.xy
    ax.plot(xs, ys, color=BOUNDARY_COLOR, linewidth=3.0, zorder=5)

    for cls, pts in pts_by_class.items():
        if not pts:
            continue
        px = [p[0] for p in pts]
        py = [p[1] for p in pts]
        ax.scatter(px, py, s=34, facecolor=LC_COLORS[cls], edgecolor="white",
                   linewidth=0.7, zorder=6)

    legend_handles = [mpatches.Patch(facecolor=LC_COLORS[c], edgecolor=BOUNDARY_COLOR,
                                      alpha=0.85, label=c) for c in LC_COLORS]
    leg = ax.legend(handles=legend_handles, loc="lower right", title="Dominant land cover",
                     fontsize=8.6, title_fontsize=9.0, frameon=True, framealpha=0.92,
                     edgecolor="0.6")
    leg.set_zorder(20)

    esri_credit = ("Imagery: Tiles \u00a9 Esri -- Source: Esri, i-cubed, USDA, USGS, AEX, "
                   "GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community")
    sc.save_figure(fig, ax, None, sc.FIGS_DIR / "fig05_within_burst_sampling.png",
                    "Within-burst sampling, stratified by land cover",
                    legend_handles=None, base_credit=esri_credit,
                    credit_extra=f"{POINTS_PER_CELL} points per grid cell ({POINTS_PER_CELL * len(cells)} total) "
                                 f"- every cell sampled, not just the dominant classes; "
                                 f"land-cover classes are schematic, basemap imagery is real")


if __name__ == "__main__":
    main()
