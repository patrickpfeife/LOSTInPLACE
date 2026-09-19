"""
Chapter-slide background: a full-bleed 16:9 image, a basemap with
several Sentinel-1/EGMS burst-frame outlines tiled over it, and a scatter of
real EGMS L2b points inside the one frame that has actual data. Purely
decorative (like fig02) - no axes, legend, colorbar or credit text baked in,
since it's meant to sit behind slide content, not stand alone as a figure.

Burst frame geometry: Sentinel-1 IW, one subswath, one burst - real dimensions
derived earlier in this project from the actual SLC annotation (range spacing
2.33 m x 21928 samples = 51.1 km; azimuth spacing 13.93 m x ~1464 lines/burst
= 20.4 km), tilted ~11 deg off north to match the real ascending track heading
(track_angle = 349 deg from the EGMS CSV used throughout this project). Only
ONE frame is real data (the Munich EGMS L2b Ascending burst used elsewhere in
presentation/); the surrounding frames are the same size/orientation tiled
next to it to illustrate "multiple burst frames," not real adjacent bursts.

Basemap provider: Esri World Street Map (server.arcgisonline.com), not CartoDB -
CartoDB's free/anonymous tiles (both Voyager and Positron) now return an
"API KEY REQUIRED" watermark on fresh requests (verified directly, not
request-pattern-specific), a live break in what worked earlier this session
for fig01/fig03 - those likely only rendered clean because those exact tiles
were already cached locally from before CARTO's tiles were gated. Esri's
World Street Map has no such gate as of this writing.

Output: presentation/figs/fig04_chapter_bg_bursts.png (1920x1080, 16:9, full-bleed)
Run:    module load uv && uv run python fig04_chapter_bg_bursts.py
"""
import zipfile
from pathlib import Path

import contextily as ctx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import affinity
from shapely.geometry import box

DATA_DIR = Path("/dss/dsshome1/0C/di54haf/test_egms/egms_density_test/data")
L2B_ASC_ZIP = DATA_DIR / "EGMS_L2b_Ascending.zip"
FIGS_DIR = Path(__file__).resolve().parent / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = FIGS_DIR / "fig04_chapter_bg_bursts.png"

CENTER_LATLON = (48.135, 11.58)   # Munich, matches the real-data crop used elsewhere
CROP_HALF_KM = 13.0               # real-data sampling box half-extent
N_POINTS_TARGET = 1300            # "some" points, not a dense mass

BURST_WIDTH_KM = 51.1             # range direction (across-track)
BURST_HEIGHT_KM = 20.4            # azimuth direction (along-track)
BURST_ROTATION_DEG = -11.0        # tilt off north, matches ascending track_angle=349 deg

FRAME_COLOR = "#0b2f6b"
FRAME_LW = 2.4
CMAP = "RdBu_r"
VLIM = 6.0

FIG_W_PX, FIG_H_PX, DPI = 1920, 1080, 160

to_3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)


def load_points(e0, n0, half_km, n_target, seed=11):
    zf = zipfile.ZipFile(L2B_ASC_ZIP)
    member = [n for n in zf.namelist() if n.endswith(".csv")][0]
    half_m = half_km * 1000
    chunks = []
    with zf.open(member) as f:
        for chunk in pd.read_csv(f, usecols=["easting", "northing", "mean_velocity"],
                                  chunksize=200_000):
            m = (chunk.easting.between(e0 - half_m, e0 + half_m) &
                 chunk.northing.between(n0 - half_m, n0 + half_m))
            if m.any():
                chunks.append(chunk[m])
    df = pd.concat(chunks, ignore_index=True)
    if len(df) > n_target:
        df = df.sample(n_target, random_state=seed)
    return df


def burst_frame(cx, cy, w_km, h_km, rot_deg):
    hw, hh = w_km * 1000 / 2, h_km * 1000 / 2
    rect = box(cx - hw, cy - hh, cx + hw, cy + hh)
    return affinity.rotate(rect, rot_deg, origin=(cx, cy))


def tiled_frames(cx, cy, w_km, h_km, rot_deg, n_across=2, n_along=3):
    """Tile frames along the burst's own azimuth (h) and range (w) axes."""
    theta = np.radians(rot_deg)
    # local "along" (height/azimuth) and "across" (width/range) unit vectors -
    # standard CCW rotation matrix [[cos,-sin],[sin,cos]] applied to (0,1) and
    # (1,0) respectively (matches shapely affinity.rotate's own convention;
    # the previous version had both signs wrong, which is what caused the
    # frames to not actually tile edge-to-edge)
    along = np.array([-np.sin(theta), np.cos(theta)])
    across = np.array([np.cos(theta), np.sin(theta)])
    frames = []
    for i in range(n_across):
        for j in range(n_along):
            off_across = (i - (n_across - 1) / 2) * w_km * 1000
            off_along = (j - (n_along - 1) / 2) * h_km * 1000
            fx = cx + across[0] * off_across + along[0] * off_along
            fy = cy + across[1] * off_across + along[1] * off_along
            frames.append(burst_frame(fx, fy, w_km, h_km, rot_deg))
    return frames


def main():
    e0, n0 = to_3035.transform(*CENTER_LATLON[::-1])
    pts = load_points(e0, n0, CROP_HALF_KM, N_POINTS_TARGET)
    print(f"loaded {len(pts)} real EGMS points")

    frames = tiled_frames(e0, n0, BURST_WIDTH_KM, BURST_HEIGHT_KM, BURST_ROTATION_DEG,
                           n_across=2, n_along=3)
    real_frame = frames[len(frames) // 2]  # the centre frame carries the real data

    allx = [x for f in frames for x, _ in f.exterior.coords]
    ally = [y for f in frames for _, y in f.exterior.coords]
    minx, maxx = min(allx), max(allx)
    miny, maxy = min(ally), max(ally)
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    w, h = maxx - minx, maxy - miny

    target_ratio = FIG_W_PX / FIG_H_PX
    margin = 1.06
    if w / h > target_ratio:
        half_w = w / 2 * margin
        half_h = half_w / target_ratio
    else:
        half_h = h / 2 * margin
        half_w = half_h * target_ratio

    fig = plt.figure(figsize=(FIG_W_PX / DPI, FIG_H_PX / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(cx - half_w, cx + half_w)
    ax.set_ylim(cy - half_h, cy + half_h)
    ax.set_aspect("equal")
    ax.set_axis_off()

    ctx.add_basemap(ax, crs="EPSG:3035", source=ctx.providers.Esri.WorldStreetMap, attribution=False)
    ax.set_xlim(cx - half_w, cx + half_w)
    ax.set_ylim(cy - half_h, cy + half_h)

    for f in frames:
        xs, ys = f.exterior.xy
        ax.plot(xs, ys, color=FRAME_COLOR, linewidth=FRAME_LW, zorder=4, solid_capstyle="round")

    ax.scatter(pts.easting, pts.northing, c=pts.mean_velocity, cmap=CMAP, vmin=-VLIM, vmax=VLIM,
               s=9, linewidths=0.2, edgecolors="white", alpha=0.9, zorder=5)

    fig.savefig(OUT_PNG, dpi=DPI)
    print(f"Saved {OUT_PNG}  ({FIG_W_PX}x{FIG_H_PX} px, 16:9)")


if __name__ == "__main__":
    main()
