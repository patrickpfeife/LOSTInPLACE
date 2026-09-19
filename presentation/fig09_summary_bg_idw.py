"""
Chapter-slide background (summary chapter): a full-bleed 16:9 image, real
EGMS L2b points over urban Munich sparsified to ~50, IDW-interpolated onto a
smooth surface, and overlaid semi-transparently on a real basemap so streets
/terrain context shows through underneath. Same decorative-background
treatment as fig02/fig04 - no axes, legend, colorbar or credit text baked in.

Basemap provider: Esri World Street Map, not literal OpenStreetMap tiles -
CartoDB's OSM-based tiles are currently gated behind an API key (discovered
and documented in fig04's docstring/build log), and raw tile.openstreetmap.org
blocks scripted/bulk fetches by policy. Esri's tiles are OSM-comparable in
content (streets, water, built-up areas) and don't require a key; same
substitution already used for fig04 and fig05.

IDW, not the GNN/kriging comparison from fig08 - this is the classical
baseline method itself, standing in as "what a simple, unlearned interpolator
of the same sparse EGMS input looks like" for a chapter-summary backdrop.

Output: presentation/figs/fig09_summary_bg_idw.png (1920x1080, 16:9, full-bleed)
Run:    module load uv && uv run python fig09_summary_bg_idw.py
"""
import zipfile
from pathlib import Path

import contextily as ctx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from pyproj import Transformer
from scipy.ndimage import gaussian_filter

DATA_DIR = Path("/dss/dsshome1/0C/di54haf/test_egms/egms_density_test/data")
L2B_ASC_ZIP = DATA_DIR / "EGMS_L2b_Ascending.zip"
FIGS_DIR = Path(__file__).resolve().parent / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = FIGS_DIR / "fig09_summary_bg_idw.png"

CENTER_LATLON = (48.135, 11.58)   # central Munich, same reference point used throughout
HALF_WIDTH_KM = 8.0               # 16 km wide - the urban area around Munich
HALF_HEIGHT_KM = 4.5              # 9 km tall -> matches 16:9 with the width above
N_POINTS = 50
SEED = 42

IDW_POWER = 2.0
GRID_W, GRID_H = 1920, 1080       # compute IDW directly at output resolution
IDW_ALPHA = 0.70
CMAP = "RdYlBu_r"  # blue->yellow->red diverging - the yellow midtone is what gives orange/yellow fringes

to_3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)


def load_sparse_points(e0, n0, half_w_m, half_h_m, n_points, seed):
    zf = zipfile.ZipFile(L2B_ASC_ZIP)
    member = [n for n in zf.namelist() if n.endswith(".csv")][0]
    chunks = []
    with zf.open(member) as f:
        for chunk in pd.read_csv(f, usecols=["easting", "northing", "mean_velocity"],
                                  chunksize=200_000):
            m = (chunk.easting.between(e0 - half_w_m, e0 + half_w_m) &
                 chunk.northing.between(n0 - half_h_m, n0 + half_h_m))
            if m.any():
                chunks.append(chunk[m])
    df = pd.concat(chunks, ignore_index=True)
    print(f"{len(df)} real EGMS points in the box; sparsifying to {n_points}")
    return df.sample(n_points, random_state=seed).reset_index(drop=True)


def idw_grid(points_xy, values, xlim, ylim, grid_w, grid_h, power=2.0):
    gx = np.linspace(xlim[0], xlim[1], grid_w)
    gy = np.linspace(ylim[0], ylim[1], grid_h)
    GX, GY = np.meshgrid(gx, gy)  # (grid_h, grid_w)

    px = points_xy[:, 0][:, None, None]
    py = points_xy[:, 1][:, None, None]
    d2 = (GX[None, :, :] - px) ** 2 + (GY[None, :, :] - py) ** 2
    # smoothing floor (~300 m, well below the ~1.7 km mean point spacing at n=50):
    # a metres-scale floor lets IDW blow up right at each sample and paints a tight
    # single-pixel "bullseye" instead of a city-scale gradient
    d2 = np.maximum(d2, 300.0 ** 2)
    w = d2 ** (-power / 2.0)
    num = (w * values[:, None, None]).sum(axis=0)
    den = w.sum(axis=0)
    return num / den, GX, GY


def main():
    e0, n0 = to_3035.transform(*CENTER_LATLON[::-1])
    half_w_m, half_h_m = HALF_WIDTH_KM * 1000, HALF_HEIGHT_KM * 1000

    pts = load_sparse_points(e0, n0, half_w_m, half_h_m, N_POINTS, SEED)
    xy = pts[["easting", "northing"]].to_numpy()
    values = pts["mean_velocity"].to_numpy()
    print(f"velocity range in sample: {values.min():.2f} .. {values.max():.2f} mm/yr")

    xlim = (e0 - half_w_m, e0 + half_w_m)
    ylim = (n0 - half_h_m, n0 + half_h_m)
    surface, GX, GY = idw_grid(xy, values, xlim, ylim, GRID_W, GRID_H, power=IDW_POWER)
    surface = gaussian_filter(surface, sigma=6)  # cohesive background gradient, not per-pixel noise

    fig = plt.figure(figsize=(GRID_W / 160, GRID_H / 160), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_axis_off()

    ctx.add_basemap(ax, crs="EPSG:3035", source=ctx.providers.Esri.WorldStreetMap, attribution=False)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # decorative background, no colourbar shown anywhere - so the diverging midpoint
    # is placed at the sample's own median rather than physical zero, purely so both
    # halves of the colour scale (blue AND yellow/orange/red) actually get used; the
    # real data skews negative, and a zero-centred scale would leave it almost all blue
    norm = TwoSlopeNorm(vmin=float(surface.min()), vcenter=float(np.median(surface)),
                         vmax=float(surface.max()))
    rgba = plt.get_cmap(CMAP)(norm(surface))
    rgba[..., 3] = IDW_ALPHA
    ax.imshow(rgba, extent=[*xlim, *ylim], origin="lower", zorder=3, interpolation="bilinear")

    ax.scatter(xy[:, 0], xy[:, 1], s=10, facecolor="none", edgecolor="white",
               linewidth=0.6, alpha=0.85, zorder=4)

    fig.savefig(OUT_PNG, dpi=160)
    print(f"Saved {OUT_PNG}  ({GRID_W}x{GRID_H} px, 16:9)")


if __name__ == "__main__":
    main()
