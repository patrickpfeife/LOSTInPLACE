"""
Sampling-strategy illustration, map 3 of 5: project.md Section 4.3 Step 2 -
random burst-anchor points with a minimum separation (Section 4.4's
neighbourhood-radius concept, illustrated as a circle around each point).
Circles cannot overlap each other, and the underlying points' own rectangle
footprint stays clear of every handpicked/GNSS rectangle (exact geometry
check, see sampling_common).

GNSS station points from map 2 are removed here; the handpicked and GNSS
rectangles stay, unlabelled, as context for where the random draw was and
wasn't allowed to land.

Output: presentation/figs/fig03_map3_random_points.png (fixed 16:9)
Run:    module load uv && uv run python fig03_map3_random_points.py
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

    xs = [p[0] for p in rand_pts]
    ys = [p[1] for p in rand_pts]
    ax.scatter(xs, ys, s=12, color=sc.COLOR_RANDOM, edgecolor="white", linewidth=0.5, zorder=5)
    for x, y in rand_pts:
        ax.add_patch(plt.Circle((x, y), sc.RANDOM_RADIUS_KM * 1000, facecolor="none",
                                 edgecolor=sc.COLOR_RANDOM, linewidth=1.2, zorder=4))

    sc.save_figure(fig, ax, countries, sc.FIGS_DIR / "fig03_map3_random_points.png",
                    "3. Random background draw, minimum-distance circles",
                    legend_handles=sc.category_legend_handles(include_random=False),
                    credit_extra=f"{len(rand_pts)} random points, "
                                 f"{sc.RANDOM_RADIUS_KM:.0f} km illustrative min-distance radius")


if __name__ == "__main__":
    main()
