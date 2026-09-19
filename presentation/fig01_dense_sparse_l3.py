"""
Figure 1 - EGMS coverage: urban vs. rural, L2b point cloud vs. L3 ortho grid.

2x2 layout for a 16:9 slide, OSM-derived basemap under everything:
  columns : Urban (Munich)          |  Rural (Bavaria, fields/forest)
  rows    : EGMS L2b (PS points)    |  EGMS L3 ortho-Up (100 m grid)

Each column uses ONE fixed AOI for both rows, so the L2b-vs-L3 comparison is
over identical ground in the same land-cover regime, not two different places.
AOI is deliberately small (~1.4 x 0.9 km) so individual PS points are visible
as discrete markers rather than merging into a solid coloured mass.

Source data (untouched, read directly from the original zips):
  test_egms/egms_density_test/data/EGMS_L2b_Ascending.zip
  test_egms/egms_density_test/data/EGMS_ortho_up.zip
    (its CSV, not the .tiff - the L3 ortho product ships BOTH a 100 m raster and
    a point table of the same cell centroids, same pid/easting/northing/mean_velocity
    schema as L2b. Plotting L3 as points-on-a-lattice, not a semi-transparent raster,
    is what actually makes 'this is a coarse fixed grid' read clearly once the AOI is
    zoomed in tight - a raster wash at this scale just looks like a smooth fill.)
Basemap: OpenStreetMap-derived cartography (CARTO Voyager), fetched live via
contextily. NOTE: tile.openstreetmap.org itself blocks automated/scripted
fetches per its tile usage policy (osm.wiki/Blocked) - hitting it directly
returns a "blocked" placeholder tile, not the map. CARTO Voyager is OSM-based
cartography (same underlying OpenStreetMap data, redrawn) served in a way that
explicitly permits this kind of programmatic use, which is why it's used here.

Output:
  presentation/figs/fig01_dense_sparse_l3.png
  presentation/figs/fig01_dense_sparse_l3.json   (every number printed on the figure)

Run:
  module load uv && uv run python fig01_dense_sparse_l3.py
"""
import json
import zipfile
from pathlib import Path

import contextily as ctx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib_scalebar.scalebar import ScaleBar
from pyproj import Transformer

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DATA_DIR = Path("/dss/dsshome1/0C/di54haf/test_egms/egms_density_test/data")
L2B_ASC_ZIP = DATA_DIR / "EGMS_L2b_Ascending.zip"
ORTHO_UP_ZIP = DATA_DIR / "EGMS_ortho_up.zip"

FIGS_DIR = Path(__file__).resolve().parent / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = FIGS_DIR / "fig01_dense_sparse_l3.png"
OUT_JSON = FIGS_DIR / "fig01_dense_sparse_l3.json"

# AOI centres, chosen from an actual point-density scan of the burst (see fig01
# dev notes): (urban) is the densest cell in central Munich; (rural) is a
# low-density cell with full (non-edge) burst coverage, well clear of the
# Munich hotspot, whose points form scattered small clusters rather than one
# road/rail alignment (checked against 4 other rural candidates).
URBAN_CENTER_LATLON = (48.133, 11.572)     # central Munich
RURAL_CENTER_LATLON = (48.1484, 12.1371)   # rural Bavaria, fields/forest

# AOI is a rectangle, not a square: wider than tall so the 2x2 grid fills a
# 16:9 slide without empty gutters, and small enough (~500 m half-extent) that
# individual PS points read as discrete markers, not a merged blob.
HALF_WIDTH_M = 700.0
HALF_HEIGHT_M = 450.0
SCALEBAR_M = 250

CMAP = "RdBu_r"

CREDIT = ("Map tiles © CARTO, map data © OpenStreetMap contributors   ·   "
          "Data: EGMS · Copernicus Land Monitoring Service · © European Union")
PRODUCT_LINE = "L2b Ascending, track 044/burst 0285, IW1 (2020-2024)  ·  L3 vertical ortho, tile E44N27_100km"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
})

to_3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
to_4326 = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)


def center_en(lat, lon):
    e, n = to_3035.transform(lon, lat)
    return e, n


def sym_vlim(values, pct=99.0, min_step=0.5):
    """Symmetric-about-zero colour limit sized to the data actually present,
    rounded up to a 'nice' step so the colourbar ticks land on clean numbers."""
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    v = float(np.percentile(np.abs(values), pct))
    step = min_step
    for candidate in (0.5, 1, 2, 2.5, 5, 10, 20, 25, 50):
        if v <= candidate:
            step = candidate
            break
    else:
        step = 50
    return max(step, np.ceil(v / step) * step)


def load_l2b_points(zip_path, e0, n0, half_w, half_h):
    """Stream the (large) L2b CSV and keep only points within the AOI box.

    Filters on easting/northing (already EPSG:3035 metres in the EGMS CSV),
    so this needs no reprojection of the points themselves.
    """
    zf = zipfile.ZipFile(zip_path)
    member = [n for n in zf.namelist() if n.endswith(".csv")][0]
    cols = ["pid", "easting", "northing", "mean_velocity"]
    chunks = []
    with zf.open(member) as f:
        for chunk in pd.read_csv(f, usecols=cols, chunksize=200_000):
            m = (chunk.easting.between(e0 - half_w, e0 + half_w) &
                 chunk.northing.between(n0 - half_h, n0 + half_h))
            if m.any():
                chunks.append(chunk[m])
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols)


def load_l3_points(zip_path, e0, n0, half_w, half_h):
    """Stream the L3 ortho CSV (cell-centroid points, same schema as L2b) and
    keep only points within the AOI box. Same filtering approach as
    load_l2b_points - L3's CSV carries pid/easting/northing/mean_velocity too."""
    zf = zipfile.ZipFile(zip_path)
    member = [n for n in zf.namelist() if n.endswith(".csv")][0]
    cols = ["pid", "easting", "northing", "mean_velocity"]
    chunks = []
    with zf.open(member) as f:
        for chunk in pd.read_csv(f, usecols=cols, chunksize=200_000):
            m = (chunk.easting.between(e0 - half_w, e0 + half_w) &
                 chunk.northing.between(n0 - half_h, n0 + half_h))
            if m.any():
                chunks.append(chunk[m])
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols)


def style_map_axis(ax, e0, n0, half_w, half_h, top_label=False, lat_side=None):
    ax.set_xlim(e0 - half_w, e0 + half_w)
    ax.set_ylim(n0 - half_h, n0 + half_h)
    ax.set_aspect("equal")

    # grid + tick labels in real lat/lon, not projected easting/northing - at
    # this AOI size (<1.5 km) the projected grid and the true graticule are
    # visually indistinguishable, so evenly-spaced ticks in metres are kept
    # for placement and simply labelled with the lat/lon they fall on.
    def lon_fmt(v, pos, n0=n0):
        lon, _ = to_4326.transform(v, n0)
        return f"{lon:.3f}°"

    def lat_fmt(v, pos, e0=e0):
        _, lat = to_4326.transform(e0, v)
        return f"{lat:.3f}°"

    ax.xaxis.set_major_formatter(FuncFormatter(lon_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(lat_fmt))
    ax.xaxis.set_major_locator(MultipleLocator(250))
    ax.yaxis.set_major_locator(MultipleLocator(250))
    ax.grid(True, which="major", linestyle=":", linewidth=0.6, color="black", alpha=0.28, zorder=5)
    ax.tick_params(labelsize=7.5, rotation=0)
    if lat_side == "left":
        ax.set_ylabel("latitude", fontsize=8.5)
        ax.yaxis.set_label_position("left")
        ax.yaxis.tick_left()
    elif lat_side == "right":
        ax.set_ylabel("latitude", fontsize=8.5)
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
    else:
        ax.tick_params(labelleft=False, labelright=False)
    if top_label:
        ax.set_xlabel("longitude", fontsize=8.5)
        ax.xaxis.set_label_position("top")
    else:
        ax.set_xlabel("longitude", fontsize=8.5)
    for spine in ax.spines.values():
        spine.set_color("0.3")
    sb = ScaleBar(1, units="m", fixed_value=SCALEBAR_M, fixed_units="m",
                  location="lower right", box_alpha=0.75,
                  font_properties={"size": 7.5}, border_pad=0.4)
    ax.add_artist(sb)


def add_basemap(ax, e0, n0, half_w, half_h):
    ax.set_xlim(e0 - half_w, e0 + half_w)
    ax.set_ylim(n0 - half_h, n0 + half_h)
    ctx.add_basemap(ax, crs="EPSG:3035", source=ctx.providers.CartoDB.Voyager,
                     attribution=False, zorder=0)
    # add_basemap can snap the extent to the fetched tile edges - re-crop to the AOI
    ax.set_xlim(e0 - half_w, e0 + half_w)
    ax.set_ylim(n0 - half_h, n0 + half_h)


def panel_tag(ax, letter):
    ax.text(0.02, 0.95, letter, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="top", ha="left", color="black",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.5", alpha=0.85))


def info_box(ax, text):
    ax.text(0.02, 0.05, text, transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.85), zorder=6)


def main():
    e_u, n_u = center_en(*URBAN_CENTER_LATLON)
    e_r, n_r = center_en(*RURAL_CENTER_LATLON)

    print("Loading urban (Munich) L2b points ...")
    pts_u = load_l2b_points(L2B_ASC_ZIP, e_u, n_u, HALF_WIDTH_M, HALF_HEIGHT_M)
    print("Loading rural (Bavaria) L2b points ...")
    pts_r = load_l2b_points(L2B_ASC_ZIP, e_r, n_r, HALF_WIDTH_M, HALF_HEIGHT_M)
    print("Loading L3 ortho points (urban AOI) ...")
    l3_u = load_l3_points(ORTHO_UP_ZIP, e_u, n_u, HALF_WIDTH_M, HALF_HEIGHT_M)
    print("Loading L3 ortho points (rural AOI) ...")
    l3_r = load_l3_points(ORTHO_UP_ZIP, e_r, n_r, HALF_WIDTH_M, HALF_HEIGHT_M)

    area_km2 = (2 * HALF_WIDTH_M / 1000) * (2 * HALF_HEIGHT_M / 1000)

    # colour scale from the data actually visible in these AOIs, not the full dataset
    vlim_l2b = sym_vlim(pd.concat([pts_u.mean_velocity, pts_r.mean_velocity]), pct=99)
    vlim_l3 = sym_vlim(pd.concat([l3_u.mean_velocity, l3_r.mean_velocity]), pct=99)

    stats = {
        "aoi_width_km": 2 * HALF_WIDTH_M / 1000, "aoi_height_km": 2 * HALF_HEIGHT_M / 1000,
        "area_km2": area_km2,
        "urban_l2b_n": len(pts_u), "urban_l2b_density_per_km2": round(len(pts_u) / area_km2, 1),
        "rural_l2b_n": len(pts_r), "rural_l2b_density_per_km2": round(len(pts_r) / area_km2, 1),
        "l2b_density_ratio_urban_over_rural": round(len(pts_u) / max(len(pts_r), 1), 1),
        "urban_l3_n": len(l3_u), "urban_l3_density_per_km2": round(len(l3_u) / area_km2, 1),
        "rural_l3_n": len(l3_r), "rural_l3_density_per_km2": round(len(l3_r) / area_km2, 1),
        "vlim_l2b_mm_yr": vlim_l2b, "vlim_l3_mm_yr": vlim_l3,
    }
    print(json.dumps(stats, indent=2))
    OUT_JSON.write_text(json.dumps(stats, indent=2))

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.0))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # (a) urban, L2b
    add_basemap(ax_a, e_u, n_u, HALF_WIDTH_M, HALF_HEIGHT_M)
    sc = ax_a.scatter(pts_u.easting, pts_u.northing, c=pts_u.mean_velocity,
                       cmap=CMAP, vmin=-vlim_l2b, vmax=vlim_l2b, s=10, linewidths=0.25,
                       edgecolors="0.15", alpha=0.85, zorder=3)
    style_map_axis(ax_a, e_u, n_u, HALF_WIDTH_M, HALF_HEIGHT_M, top_label=True, lat_side="left")
    panel_tag(ax_a, "a")
    info_box(ax_a, f"n = {len(pts_u):,}\n{stats['urban_l2b_density_per_km2']:,.0f} pts/km²")

    # (b) rural, L2b
    add_basemap(ax_b, e_r, n_r, HALF_WIDTH_M, HALF_HEIGHT_M)
    ax_b.scatter(pts_r.easting, pts_r.northing, c=pts_r.mean_velocity,
                 cmap=CMAP, vmin=-vlim_l2b, vmax=vlim_l2b, s=22, linewidths=0.35,
                 edgecolors="0.15", alpha=0.9, zorder=3)
    style_map_axis(ax_b, e_r, n_r, HALF_WIDTH_M, HALF_HEIGHT_M, top_label=True, lat_side="right")
    panel_tag(ax_b, "b")
    info_box(ax_b, f"n = {len(pts_r):,}\n{stats['rural_l2b_density_per_km2']:,.0f} pts/km²")

    # (c) urban, L3 - same AOI as (a). Exact same marker style as the L2b row
    # (circles, same size/edge/alpha per column) - no visual distinction beyond
    # what the data itself shows (the fixed 100 m spacing is obvious on its own).
    add_basemap(ax_c, e_u, n_u, HALF_WIDTH_M, HALF_HEIGHT_M)
    sc_l3 = ax_c.scatter(l3_u.easting, l3_u.northing, c=l3_u.mean_velocity,
                          cmap=CMAP, vmin=-vlim_l3, vmax=vlim_l3, s=10, linewidths=0.25,
                          edgecolors="0.15", alpha=0.85, zorder=3)
    style_map_axis(ax_c, e_u, n_u, HALF_WIDTH_M, HALF_HEIGHT_M, top_label=False, lat_side="left")
    panel_tag(ax_c, "c")
    info_box(ax_c, f"n = {len(l3_u):,}\n{stats['urban_l3_density_per_km2']:,.0f} pts/km²")

    # (d) rural, L3 - same AOI as (b)
    add_basemap(ax_d, e_r, n_r, HALF_WIDTH_M, HALF_HEIGHT_M)
    ax_d.scatter(l3_r.easting, l3_r.northing, c=l3_r.mean_velocity,
                 cmap=CMAP, vmin=-vlim_l3, vmax=vlim_l3, s=22, linewidths=0.35,
                 edgecolors="0.15", alpha=0.9, zorder=3)
    style_map_axis(ax_d, e_r, n_r, HALF_WIDTH_M, HALF_HEIGHT_M, top_label=False, lat_side="right")
    panel_tag(ax_d, "d")
    info_box(ax_d, f"n = {len(l3_r):,}\n{stats['rural_l3_density_per_km2']:,.0f} pts/km²")

    fig.subplots_adjust(left=0.055, right=0.885, top=0.86, bottom=0.145, wspace=0.13, hspace=0.20)

    # column headers (span each column, above the top row's own top-axis tick labels)
    for ax, label in [(ax_a, "Urban -- Munich"), (ax_b, "Rural -- Bavaria (fields / forest)")]:
        pos = ax.get_position()
        fig.text((pos.x0 + pos.x1) / 2, 0.975, label, ha="center", va="bottom",
                  fontsize=13, fontweight="bold")

    # row labels (rotated, left of each row)
    for ax, label in [(ax_a, "EGMS L2b\n(PS points)"), (ax_c, "EGMS L3 ortho-Up\n(100 m grid)")]:
        pos = ax.get_position()
        fig.text(0.014, (pos.y0 + pos.y1) / 2, label, ha="left", va="center",
                  fontsize=10.5, fontweight="bold", rotation=90, linespacing=1.3)

    # colorbars, one per row, aligned to that row's vertical extent, scaled to
    # that row's own data-driven limit (not a fixed dataset-wide guess)
    pa, pc = ax_a.get_position(), ax_c.get_position()
    cax_top = fig.add_axes([0.905, pa.y0, 0.016, pa.y1 - pa.y0])
    cb_top = fig.colorbar(sc, cax=cax_top)
    cb_top.ax.set_title("LOS vel.\n[mm/yr]", fontsize=8, pad=7, linespacing=1.2)
    cb_top.ax.tick_params(labelsize=7.5)

    cax_bot = fig.add_axes([0.905, pc.y0, 0.016, pc.y1 - pc.y0])
    cb_bot = fig.colorbar(sc_l3, cax=cax_bot)
    cb_bot.ax.set_title("Up vel.\n[mm/yr]", fontsize=8, pad=7, linespacing=1.2)
    cb_bot.ax.tick_params(labelsize=7.5)

    fig.text(0.055, 0.060, PRODUCT_LINE, fontsize=7.5, color="0.3", ha="left")
    fig.text(0.055, 0.030, CREDIT, fontsize=7.5, color="0.3", ha="left")

    fig.savefig(OUT_PNG, dpi=300)
    print(f"\nSaved {OUT_PNG}")


if __name__ == "__main__":
    main()
