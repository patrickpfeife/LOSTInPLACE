"""
Sampling-strategy illustration, map 2 of 5: adds the GNSS validation network
on top of map 1 (project.md Section 4.1: "Reserve validation regions with
GNSS"). Station positions are a schematic scatter, not real EPN/EUREF
coordinates. Every GNSS rectangle is placed to not overlap the handpicked
rectangles or each other (exact geometry check, see sampling_common).

Output: presentation/figs/fig03_map2_gnss_stations.png (fixed 16:9)
Run:    module load uv && uv run python fig03_map2_gnss_stations.py
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

    fig, ax = sc.setup_figure()
    sc.draw_countries(ax, countries)
    sc.draw_rects(ax, [g for _, _, g, _ in handpicked], sc.COLOR_HANDPICKED)
    sc.draw_rects(ax, gnss_rects, sc.COLOR_GNSS)
    xs = [p[0] for p in gnss_pts]
    ys = [p[1] for p in gnss_pts]
    ax.scatter(xs, ys, s=16, color=sc.COLOR_GNSS, edgecolor="white", linewidth=0.6, zorder=5)

    sc.save_figure(fig, ax, countries, sc.FIGS_DIR / "fig03_map2_gnss_stations.png",
                    "2. GNSS validation network reserved",
                    legend_handles=sc.category_legend_handles(include_random=False),
                    credit_extra=f"{len(gnss_pts)} schematic GNSS stations")


if __name__ == "__main__":
    main()
