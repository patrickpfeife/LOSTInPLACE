"""
Shared building blocks for the fig03_map*.py series - 5 maps illustrating the
project.md Section 4.3 sampling strategy, each map adding one layer on top of
the last. Not a plot itself; imported by fig03_map{1..5}_*.py.

Everything here is deterministic (fixed RNG seeds) so that independently
running fig03_map3 / map4 / map5 regenerates the *same* GNSS stations and
random background points each time - required since map4/5 reuse exactly the
points map3 draws, and each script is a standalone, separately-runnable file
per this project's convention (no shared cache file to keep in sync instead).

NON-OVERLAP: every rectangle (handpicked, GNSS, random) is placed by exact
shapely geometry intersection tests against every rectangle already placed
(not an approximate min-centre-distance heuristic) - required so that no two
rectangles ever overlap, regardless of their relative size or the angle
between them.

Visual encoding (see chat discussion): FILL colour = which step/process
produced the region (handpicked / GNSS / random background). BOUNDARY colour
= which split it belongs to (train / dev / validation) - only meaningful from
map 4/5 onward, since the split isn't assigned yet in maps 1-3. GNSS regions
are always "validation" (project.md Section 8.1 Tier 2: product-level
validation happens at GNSS sites); 2 of the 6 handpicked regions are also
reserved as validation (project.md Section 4.3: "reserve some handpicked
regions entirely for test") - Firenze-Prato-Pistoia (explicitly the "optional
2nd" groundwater example - project.md's own stated point of it is testing
"whether the model learned the physics or memorised the place", which is
exactly what a held-out test region is for) and Green Heart (a natural
uncertainty-calibration stress test to hold out, given its role in project.md
Section 10.4).

Country boundaries: Natural Earth 1:50m Admin-0 Countries (public domain),
downloaded once to presentation/data/ne_50m_admin_0_countries/.
EGMS coverage classification (27 EU states + UK + Norway + Iceland) per
Copernicus Land Monitoring Service's stated EGMS coverage - this is a
schematic mainland classification for the illustration; overseas territories
that are technically also covered (Azores, Madeira, Canaries, French DROMs)
are outside this map's crop and not shown.
"""
from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import affinity
from shapely.geometry import Point, box

PRESENTATION_DIR = Path(__file__).resolve().parent
FIGS_DIR = PRESENTATION_DIR / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)
COUNTRIES_SHP = PRESENTATION_DIR / "data" / "ne_50m_admin_0_countries" / "ne_50m_admin_0_countries.shp"

CRS_EQAREA = "EPSG:3035"  # ETRS89-LAEA - same convention project.md uses throughout

EGMS_COUNTRIES = {
    # 27 EU member states
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czechia", "Denmark",
    "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Ireland",
    "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland",
    "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden",
    # + UK, Norway, Iceland
    "United Kingdom", "Norway", "Iceland",
}

CROP_LONLAT = (-25.0, 34.0, 35.0, 71.5)  # (minx, miny, maxx, maxy)

COLOR_LAND_EGMS = "#c9cdd2"
COLOR_LAND_NON_EGMS = "#8f959c"
COLOR_BORDER = "#ffffff"

# CATEGORY colour = which step/process produced the region. Bold, well-separated
# hues, used consistently for rectangle fill (translucent) + edge (solid) in
# every map, and for the map-5 split markers' colour too - colour always means
# "category", never "split", anywhere in the series.
COLOR_HANDPICKED = "#8c1c3f"   # bordeaux red
COLOR_GNSS = "#22b14c"         # bright green
COLOR_RANDOM = "#1f6fd1"       # blue
FILL_ALPHA = 0.30
EDGE_LW = 1.8

# SPLIT is encoded by MARKER SHAPE in map 5 (never colour): square = train,
# triangle = dev, circle = validation.
SHAPE_TRAIN = "s"
SHAPE_DEV = "^"
SHAPE_VALIDATION = "o"

# Fixed canvas: exact 16:9, matches a standard PowerPoint widescreen slide
# (13.333 x 7.5 in). Every map saves at this same size - no bbox_inches="tight",
# which was the actual cause of the earlier maps' inconsistent, mostly-blank
# canvases (it silently grew the saved canvas to fit an unwrapped credit
# line instead of wrapping the text within a fixed width).
FIG_SIZE = (13.333, 7.5)
SAVE_DPI = 220
AXES_RECT = (0.02, 0.085, 0.96, 0.735)  # left, bottom, width, height, figure-fraction
TITLE_Y = 0.925
DISCLAIMER_Y = 0.978

CREDIT = ("Natural Earth (public domain) boundaries · EGMS coverage per Copernicus Land "
          "Monitoring Service · GNSS/sample locations are schematic, not real coordinates")
DISCLAIMER = "Illustrative only — not the final sampling locations."

# ----------------------------------------------------------------------------
# project.md Section 4.3 Step 1: handpicked known-process regions.
# (label, process, lon, lat, half_width_km, half_height_km, split)
# ----------------------------------------------------------------------------
HANDPICKED_REGIONS = [
    ("Lorca (Alto Guadalentín)", "Groundwater subsidence", -1.70, 37.67, 22, 16, "train"),
    ("Upper Silesian Basin", "Mining + rebound", 18.70, 50.25, 38, 28, "train"),
    ("Green Heart, NL", "Peat (low-coherence)", 4.70, 52.05, 32, 26, "validation"),
    ("Po Delta / Ravenna / Venice", "Coastal / delta", 12.35, 44.95, 55, 30, "train"),
    ("Firenze-Prato-Pistoia", "Groundwater (2nd)", 11.05, 43.85, 26, 18, "validation"),
    ("Fennoscandian Shield", "Stable baseline (GIA)", 15.00, 62.00, 80, 80, "train"),
]

GNSS_HALF_KM = (45, 38)
RANDOM_HALF_KM = (45, 38)
RANDOM_RADIUS_KM = 55.0  # min-distance illustration circle (map 3 only)

N_GNSS = 26
N_RANDOM = 55
TRAIN_FRACTION = 0.80

SEED_GNSS = 20260826
SEED_RANDOM = 7042026
SEED_SPLIT = 314159


def load_europe(crs=CRS_EQAREA):
    gdf = gpd.read_file(COUNTRIES_SHP)
    minx, miny, maxx, maxy = CROP_LONLAT
    gdf = gdf.clip(box(minx, miny, maxx, maxy))
    gdf["egms"] = gdf["NAME"].isin(EGMS_COUNTRIES)
    gdf = gdf.to_crs(crs)
    return gdf


def region_rect_latlon(lon, lat, half_w_km, half_h_km, crs):
    pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(crs).iloc[0]
    return box(pt.x - half_w_km * 1000, pt.y - half_h_km * 1000,
               pt.x + half_w_km * 1000, pt.y + half_h_km * 1000)


def handpicked_geoms(crs=CRS_EQAREA):
    """List of (label, process, geometry, split) for the 6 handpicked regions."""
    out = []
    for label, process, lon, lat, hw, hh, split in HANDPICKED_REGIONS:
        out.append((label, process, region_rect_latlon(lon, lat, hw, hh, crs), split))
    # safety check: the hand-placed regions themselves must not overlap either
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            if out[i][2].intersects(out[j][2]):
                raise RuntimeError(f"handpicked regions overlap: {out[i][0]} / {out[j][0]}")
    return out


def _land_union(countries):
    return countries[countries.egms].geometry.union_all()


def sample_rects_in_land(land_geom, n, seed, half_w_km, half_h_km, occupied,
                          bounds=None, max_tries=500_000,
                          extra_point_check=None):
    """Rejection-sample n rectangle centres inside land_geom. Each accepted
    rectangle is checked by EXACT geometry intersection against every
    geometry already in `occupied` (a list, mutated in place - so points
    accepted earlier in this same call also exclude later ones). Guarantees
    zero rectangle overlap regardless of size/angle, unlike a min-distance
    heuristic.

    extra_point_check(pt, accepted_pts) -> bool, if given, is an additional
    accept/reject test evaluated against the (x,y) tuples accepted so far in
    THIS call (used for the map-3 circle non-overlap constraint).
    """
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = bounds if bounds is not None else land_geom.bounds
    hw_m, hh_m = half_w_km * 1000, half_h_km * 1000
    accepted_pts = []
    tries = 0
    while len(accepted_pts) < n and tries < max_tries:
        tries += 1
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        p = Point(x, y)
        if not land_geom.contains(p):
            continue
        rect = box(x - hw_m, y - hh_m, x + hw_m, y + hh_m)
        if any(rect.intersects(g) for g in occupied):
            continue
        if extra_point_check is not None and not extra_point_check((x, y), accepted_pts):
            continue
        accepted_pts.append((x, y))
        occupied.append(rect)
    if len(accepted_pts) < n:
        raise RuntimeError(f"only placed {len(accepted_pts)}/{n} points in {tries} tries")
    return accepted_pts


def gnss_stations(countries, handpicked_rects, occupied=None):
    """Returns (gnss_points, gnss_rects, occupied_list). `occupied` may be
    passed in to continue accumulating into a shared exclusion list; if None,
    a fresh one seeded with the handpicked rectangles is used."""
    land = _land_union(countries)
    if occupied is None:
        occupied = [g for _, _, g, _ in handpicked_rects]
    pts = sample_rects_in_land(land, N_GNSS, SEED_GNSS, *GNSS_HALF_KM, occupied)
    rects = occupied[-N_GNSS:]
    return pts, rects, occupied


def random_points(countries, occupied, existing_random_pts=None):
    """occupied must already contain handpicked + GNSS rectangles (mutated
    in place, GNSS rects included). Also enforces mutual circle non-overlap
    (map-3 illustration) via extra_point_check."""
    land = _land_union(countries)
    radius_m = RANDOM_RADIUS_KM * 1000

    def circles_dont_overlap(pt, accepted_pts):
        x, y = pt
        return all((x - ax) ** 2 + (y - ay) ** 2 >= (2 * radius_m) ** 2 for ax, ay in accepted_pts)

    pts = sample_rects_in_land(land, N_RANDOM, SEED_RANDOM, *RANDOM_HALF_KM, occupied,
                                extra_point_check=circles_dont_overlap)
    rects = occupied[-N_RANDOM:]
    return pts, rects


def build_all_layers(countries):
    """Convenience one-shot: returns a dict with every layer, all mutually
    non-overlapping, ready for any of the 5 maps to pick and choose from."""
    handpicked = handpicked_geoms()
    occupied = [g for _, _, g, _ in handpicked]
    gnss_pts, gnss_rects, occupied = gnss_stations(countries, handpicked, occupied)
    rand_pts, rand_rects = random_points(countries, occupied)
    is_train = train_dev_split(rand_pts)
    return dict(handpicked=handpicked, gnss_pts=gnss_pts, gnss_rects=gnss_rects,
                rand_pts=rand_pts, rand_rects=rand_rects, is_train=is_train)


def rotated_axes(rot_deg):
    """Unit vectors for a rectangle's own local (along, across) axes after
    rotating by rot_deg (degrees, counter-clockwise, matching shapely's
    affinity.rotate convention). along = local "height"/azimuth axis,
    across = local "width"/range axis."""
    theta = np.radians(rot_deg)
    along = np.array([-np.sin(theta), np.cos(theta)])
    across = np.array([np.cos(theta), np.sin(theta)])
    return along, across


def rotated_rect(cx, cy, w_km, h_km, rot_deg):
    hw, hh = w_km * 1000 / 2, h_km * 1000 / 2
    rect = box(cx - hw, cy - hh, cx + hw, cy + hh)
    return affinity.rotate(rect, rot_deg, origin=(cx, cy))


def rotated_grid(cx, cy, w_km, h_km, rot_deg, n_across, n_along):
    """n_across x n_along equal sub-cells tiling a w_km x h_km rectangle
    centred on (cx,cy) and rotated rot_deg, aligned to the rectangle's own
    axes (not lat/lon) - i.e. the same local grid a real SAR product's range
    /azimuth geometry would naturally have. Returns a flat list of polygons,
    row-major (across-major, then along)."""
    along, across = rotated_axes(rot_deg)
    cell_w, cell_h = w_km / n_across, h_km / n_along
    cells = []
    for i in range(n_across):
        for j in range(n_along):
            off_across = (i - (n_across - 1) / 2) * cell_w * 1000
            off_along = (j - (n_along - 1) / 2) * cell_h * 1000
            fx = cx + across[0] * off_across + along[0] * off_along
            fy = cy + across[1] * off_across + along[1] * off_along
            cells.append(rotated_rect(fx, fy, cell_w, cell_h, rot_deg))
    return cells


def train_dev_split(rand_pts):
    rng = np.random.default_rng(SEED_SPLIT)
    idx = np.arange(len(rand_pts))
    rng.shuffle(idx)
    n_train = int(round(TRAIN_FRACTION * len(rand_pts)))
    train_idx = set(idx[:n_train].tolist())
    return [i in train_idx for i in range(len(rand_pts))]


def style_axis(ax, countries):
    ax.set_aspect("equal")
    ax.set_axis_off()
    minx, miny, maxx, maxy = countries[countries.egms].total_bounds
    pad = 120_000
    ax.set_xlim(minx - pad, maxx + pad * 1.3)
    ax.set_ylim(miny - pad, maxy + pad)


def draw_countries(ax, countries):
    countries.plot(ax=ax, color=np.where(countries.egms, COLOR_LAND_EGMS, COLOR_LAND_NON_EGMS),
                    edgecolor=COLOR_BORDER, linewidth=0.5)


def draw_rects(ax, geoms, color, zorder=4):
    """Category rectangle: translucent fill + solid edge, both `color`."""
    gpd.GeoSeries(geoms).plot(ax=ax, facecolor=color, edgecolor=color,
                               alpha=FILL_ALPHA, linewidth=EDGE_LW, zorder=zorder)
    # alpha above also fades the edge; redraw a fully-opaque edge on top
    gpd.GeoSeries(geoms).plot(ax=ax, facecolor="none", edgecolor=color,
                               linewidth=EDGE_LW, zorder=zorder + 1)


def draw_split_markers(ax, geoms, color, shape, size=140, zorder=6):
    xs = [g.centroid.x for g in geoms]
    ys = [g.centroid.y for g in geoms]
    ax.scatter(xs, ys, marker=shape, s=size, facecolor=color, edgecolor="white",
               linewidth=0.9, zorder=zorder)


def setup_figure():
    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_axes(AXES_RECT)
    return fig, ax


def save_figure(fig, ax, countries, out_path, title, legend_handles=None,
                 legend_loc="lower right", credit_extra=None, base_credit=None):
    if countries is not None:
        style_axis(ax, countries)
    else:
        ax.set_axis_off()
    if legend_handles:
        ax.legend(handles=legend_handles, loc=legend_loc, fontsize=8.6,
                   frameon=True, framealpha=0.92, edgecolor="0.6")
    # title and disclaimer as independently-positioned fig-level text (not
    # ax.set_title, whose placement relative to the axes varies with title
    # length/wrapping and previously collided with the disclaimer on longer
    # titles) - fixed, well-separated y-coordinates guarantee no overlap.
    fig.text(0.5, TITLE_Y, title, fontsize=14, fontweight="bold", ha="center", va="center")
    fig.text(0.5, DISCLAIMER_Y, DISCLAIMER, fontsize=9.5, color="#8a1414", ha="center",
              va="top", fontstyle="italic", fontweight="bold")
    base = CREDIT if base_credit is None else base_credit
    credit = base if credit_extra is None else base + "  ·  " + credit_extra
    fig.text(0.5, 0.028, credit, fontsize=8.6, color="0.25", ha="center", va="center",
              wrap=True)
    fig.savefig(out_path, dpi=SAVE_DPI)
    print(f"Saved {out_path}  ({FIG_SIZE[0]}x{FIG_SIZE[1]} in, 16:9)")


def category_legend_handles(include_gnss=True, include_random=True):
    handles = [mpatches.Patch(facecolor=COLOR_HANDPICKED, edgecolor=COLOR_HANDPICKED,
                               alpha=FILL_ALPHA + 0.15, label="Handpicked region")]
    if include_gnss:
        handles.append(mpatches.Patch(facecolor=COLOR_GNSS, edgecolor=COLOR_GNSS,
                                       alpha=FILL_ALPHA + 0.15, label="GNSS validation region"))
    if include_random:
        handles.append(mpatches.Patch(facecolor=COLOR_RANDOM, edgecolor=COLOR_RANDOM,
                                       alpha=FILL_ALPHA + 0.15, label="Random background region"))
    return handles


def split_legend_handles(n_train=None, n_dev=None, n_val=None):
    def lbl(base, n):
        return base if n is None else f"{base} ({n})"
    grey = "0.25"
    return [
        mlines.Line2D([0], [0], marker=SHAPE_TRAIN, color="none", markerfacecolor=grey,
                      markeredgecolor="white", markersize=10, label=lbl("Train", n_train)),
        mlines.Line2D([0], [0], marker=SHAPE_DEV, color="none", markerfacecolor=grey,
                      markeredgecolor="white", markersize=11, label=lbl("Dev", n_dev)),
        mlines.Line2D([0], [0], marker=SHAPE_VALIDATION, color="none", markerfacecolor=grey,
                      markeredgecolor="white", markersize=10, label=lbl("Validation", n_val)),
    ]
