"""
Sampling-strategy illustration, map 1 of 5: project.md Section 4.3 Step 1 -
the six handpicked known-process regions, labelled. Colour = process category
(constant across the whole 5-map series; only map 5 additionally encodes the
train/dev/validation split, via marker shape, not colour).

Output: presentation/figs/fig03_map1_handpicked_regions.png (fixed 16:9)
Run:    module load uv && uv run python fig03_map1_handpicked_regions.py
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

    fig, ax = sc.setup_figure()
    sc.draw_countries(ax, countries)
    sc.draw_rects(ax, [g for _, _, g, _ in handpicked], sc.COLOR_HANDPICKED)

    label_overrides = {
        "Po Delta / Ravenna / Venice": dict(xytext=(70, 10), ha="left", va="center"),
        "Firenze-Prato-Pistoia": dict(xytext=(-70, -55), ha="right", va="center"),
    }
    for label, process, geom, _split in handpicked:
        cx, cy = geom.centroid.x, geom.centroid.y
        miny = geom.bounds[1]
        opts = label_overrides.get(label)
        if opts is None:
            xy, xytext, ha, va = (cx, miny), (0, -8), "center", "top"
        else:
            xy, xytext, ha, va = (cx, cy), opts["xytext"], opts["ha"], opts["va"]
        ax.annotate(f"{label}\n({process})", xy=xy, xytext=xytext,
                    textcoords="offset points", ha=ha, va=va,
                    fontsize=8.3, color=sc.COLOR_HANDPICKED, fontweight="bold",
                    linespacing=1.3,
                    arrowprops=dict(arrowstyle="-", color=sc.COLOR_HANDPICKED, lw=0.7)
                    if opts is not None else None)

    sc.save_figure(fig, ax, countries, sc.FIGS_DIR / "fig03_map1_handpicked_regions.png",
                    "1. Handpicked known-process regions",
                    legend_handles=sc.category_legend_handles(include_gnss=False, include_random=False))


if __name__ == "__main__":
    main()
