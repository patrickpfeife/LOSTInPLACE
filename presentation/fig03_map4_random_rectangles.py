"""
Sampling-strategy illustration, map 4 of 5: same layers as map 2 (handpicked
+ GNSS), but the map-3 random points and their min-distance circles are now
formalised as random-category rectangles - same centres as map 3's accepted
draw. All three categories are mutually non-overlapping by construction
(see sampling_common).

Output: presentation/figs/fig03_map4_random_rectangles.png (fixed 16:9)
Run:    module load uv && uv run python fig03_map4_random_rectangles.py
"""
import matplotlib.pyplot as plt

import sampling_common as sc

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
})


def main():
    countries = sc.load_europe()
    handpicked = sc.handpicked_geoms()
    occupied = [g for _, _, g, _ in handpicked]
    gnss_pts, gnss_rects, occupied = sc.gnss_stations(countries, handpicked, occupied)
    rand_pts, rand_rects = sc.random_points(countries, occupied)

    fig, ax = sc.setup_figure()
    sc.draw_countries(ax, countries)
    sc.draw_rects(ax, [g for _, _, g, _ in handpicked], sc.COLOR_HANDPICKED)
    sc.draw_rects(ax, gnss_rects, sc.COLOR_GNSS)
    sc.draw_rects(ax, rand_rects, sc.COLOR_RANDOM)

    sc.save_figure(fig, ax, countries, sc.FIGS_DIR / "fig03_map4_random_rectangles.png",
                    "4. Random draw formalised as burst-anchor regions",
                    legend_handles=sc.category_legend_handles(),
                    credit_extra=f"{len(rand_rects)} random regions; all {len(occupied)} "
                                 f"regions shown are mutually non-overlapping")


if __name__ == "__main__":
    main()
