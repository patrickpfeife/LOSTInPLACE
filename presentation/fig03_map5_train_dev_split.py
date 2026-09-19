"""
Sampling-strategy illustration, map 5 of 5: the full dual encoding.

IMPORTANT DIFFERENCE FROM MAPS 1-4: this map does NOT draw the actual AOI
rectangles. Earlier versions drew the rectangle footprint (fill+edge) AND a
separate shape marker on top of it at the same spot - two overlapping glyphs
per region, which is exactly what produced the "overlapping symbols" problem.
The fix is structural, not cosmetic: each region gets exactly ONE glyph here,
whose SHAPE encodes the split and whose COLOUR encodes the category -
- Train -> square, Dev -> triangle, Validation -> circle (never colour).
- Handpicked -> bordeaux red, GNSS -> bright green, Random -> blue (never shape).
Regions were already placed with generous mutual separation as non-overlapping
rectangles (see sampling_common); a single, modestly-sized marker at each
centroid inherits that same clearance, so the glyphs don't collide either.

  - GNSS regions -> always validation (project.md Section 8.1 Tier 2).
  - 2 of 6 handpicked regions -> validation (project.md Section 4.3: "reserve
    some handpicked regions entirely for test"): Green Heart (uncertainty
    stress-test role, Section 10.4) and Firenze-Prato-Pistoia (explicitly the
    "optional 2nd" groundwater example, whose stated point is testing
    "whether the model learned the physics or memorised the place").
  - Random regions -> 80% train / 20% dev (project.md Section 4.5).

Output: presentation/figs/fig03_map5_train_dev_split.png (fixed 16:9)
Run:    module load uv && uv run python fig03_map5_train_dev_split.py
"""
import matplotlib.pyplot as plt

import sampling_common as sc

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
})

MARKER_SIZE = 170


def main():
    countries = sc.load_europe()
    handpicked = sc.handpicked_geoms()
    occupied = [g for _, _, g, _ in handpicked]
    gnss_pts, gnss_rects, occupied = sc.gnss_stations(countries, handpicked, occupied)
    rand_pts, rand_rects = sc.random_points(countries, occupied)
    is_train = sc.train_dev_split(rand_pts)

    fig, ax = sc.setup_figure()
    sc.draw_countries(ax, countries)

    # exactly one glyph per region: shape = split, colour = category
    hp_train = [g for _, _, g, s in handpicked if s == "train"]
    hp_val = [g for _, _, g, s in handpicked if s == "validation"]
    sc.draw_split_markers(ax, hp_train, sc.COLOR_HANDPICKED, sc.SHAPE_TRAIN, size=MARKER_SIZE)
    sc.draw_split_markers(ax, hp_val, sc.COLOR_HANDPICKED, sc.SHAPE_VALIDATION, size=MARKER_SIZE)
    sc.draw_split_markers(ax, gnss_rects, sc.COLOR_GNSS, sc.SHAPE_VALIDATION, size=MARKER_SIZE)

    train_rects = [b for b, t in zip(rand_rects, is_train) if t]
    dev_rects = [b for b, t in zip(rand_rects, is_train) if not t]
    sc.draw_split_markers(ax, train_rects, sc.COLOR_RANDOM, sc.SHAPE_TRAIN, size=MARKER_SIZE)
    sc.draw_split_markers(ax, dev_rects, sc.COLOR_RANDOM, sc.SHAPE_DEV, size=MARKER_SIZE)

    n_hp_val = len(hp_val)
    n_hp_train = len(hp_train)
    n_val = n_hp_val + len(gnss_rects)
    n_train = n_hp_train + len(train_rects)
    n_dev = len(dev_rects)

    cat_legend = ax.legend(handles=sc.category_legend_handles(), loc="lower left",
                            title="Colour = category", fontsize=8.3, title_fontsize=8.8,
                            frameon=True, framealpha=0.96, edgecolor="0.6")
    cat_legend.set_zorder(20)
    ax.add_artist(cat_legend)
    split_legend = ax.legend(
        handles=sc.split_legend_handles(n_train=n_train, n_dev=n_dev, n_val=n_val),
        loc="lower right", title="Shape = set", fontsize=8.3, title_fontsize=8.8,
        frameon=True, framealpha=0.96, edgecolor="0.6")
    split_legend.set_zorder(20)

    sc.save_figure(fig, ax, countries, sc.FIGS_DIR / "fig03_map5_train_dev_split.png",
                    "5. Train / dev / validation split",
                    legend_handles=None,  # legends already added above
                    credit_extra="validation = all GNSS + 2 handpicked regions "
                                 "(Green Heart, Firenze-Prato-Pistoia)")
    print(f"train={n_train} dev={n_dev} validation={n_val}")


if __name__ == "__main__":
    main()
