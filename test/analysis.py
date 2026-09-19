"""
EGMS L3 ortho (vertical + east-west) vs NGL GNSS comparison, Munich area.

Reads:
  - data/OBE4.EU.tenv3, data/D256.EU.tenv3   (NGL, Eurasia-fixed frame)
  - data/OBE4.IGS20.tenv3, data/D256.IGS20.tenv3 (NGL, ITRF/global frame, for the plate-motion illustration)
  - data/egms_matched_up.csv, data/egms_matched_east.csv (nearest EGMS pixel per station, from extract_egms_pixels.py)

Writes:
  - figs/*.png
  - data/comparison_results.csv  (the numbers that go into report.md)
"""
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DATA = "/dss/dsshome1/0C/di54haf/LOSTInPLACE-Line-Of-Sight-To-INSAR-derived-continuous-DisPLACEment/test/data"
FIGS = "/dss/dsshome1/0C/di54haf/LOSTInPLACE-Line-Of-Sight-To-INSAR-derived-continuous-DisPLACEment/test/figs"

# ---- dataviz palette (light mode) ----
C_SURFACE = "#fcfcfb"
C_PRIMARY = "#0b0b0b"
C_SECONDARY = "#52514e"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_BASELINE = "#c3c2b7"
C_BLUE = "#2a78d6"    # EGMS / EU-fixed (correct)
C_ORANGE = "#eb6834"  # GNSS
C_RED = "#e34948"     # IGS20 (plate-motion contaminated)

plt.rcParams.update({
    "figure.facecolor": C_SURFACE,
    "axes.facecolor": C_SURFACE,
    "axes.edgecolor": C_BASELINE,
    "axes.labelcolor": C_PRIMARY,
    "text.color": C_PRIMARY,
    "xtick.color": C_MUTED,
    "ytick.color": C_MUTED,
    "grid.color": C_GRID,
    "font.family": "sans-serif",
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})

TENV3_COLS = ["site", "yymmmdd", "decyear", "mjd", "week", "d", "reflon",
              "e0", "east_m", "n0", "north_m", "u0", "up_m", "ant",
              "sig_e", "sig_n", "sig_u", "corr_en", "corr_eu", "corr_nu",
              "lat", "lon", "height"]


def flag_outliers(series, window=15, thresh=3.5):
    """Modified Z-score outlier flag (Iglewicz & Hoaglin) against a rolling median.
    Returns a boolean mask, True = outlier. thresh=3.5 is the standard default."""
    med = series.rolling(window, center=True, min_periods=5).median()
    resid = series - med
    mad = np.median(np.abs(resid.dropna()))
    if mad == 0 or np.isnan(mad):
        return pd.Series(False, index=series.index)
    modz = 0.6745 * resid / mad
    return modz.abs() > thresh


def load_tenv3(path, despike=True):
    df = pd.read_csv(path, sep=r"\s+", skiprows=1, header=None, names=TENV3_COLS)
    df["date"] = pd.to_datetime(df["yymmmdd"], format="%y%b%d")
    df["east_mm"] = df["east_m"] * 1000.0
    df["north_mm"] = df["north_m"] * 1000.0
    df["up_mm"] = df["up_m"] * 1000.0
    df = df.sort_values("date").reset_index(drop=True)
    if despike:
        # Raw NGL daily solutions contain a small fraction of clearly bad epochs
        # (single-day jumps of tens of mm against ~5mm formal sigma - antenna/
        # multipath/processing artifacts, not real motion). We flag and drop
        # these with a standard robust outlier test (modified Z-score, window=15
        # obs, threshold=3.5) applied independently to east/north/up, uniformly
        # for every station and component before any comparison is made.
        bad = pd.Series(False, index=df.index)
        for col in ["east_mm", "north_mm", "up_mm"]:
            bad = bad | flag_outliers(df[col])
        n_bad = bad.sum()
        if n_bad:
            print(f"  [{path.split('/')[-1]}] dropping {n_bad}/{len(df)} outlier epochs (modified Z-score > 3.5)")
        df = df.loc[~bad].reset_index(drop=True)
    return df


def load_egms_pixel(matched_csv, station):
    df = pd.read_csv(matched_csv)
    row = df[df["station"] == station].iloc[0]
    date_cols = [c for c in df.columns if c.isdigit() and len(c) == 8]
    dates = pd.to_datetime(date_cols, format="%Y%m%d")
    values = row[date_cols].to_numpy(dtype=float)
    ts = pd.DataFrame({"date": dates, "value_mm": values}).sort_values("date").reset_index(drop=True)
    meta = {k: row[k] for k in ["pid", "easting", "northing", "dist_to_station_m",
                                 "mean_velocity", "gnss_velocity_u", "gnss_velocity_e",
                                 "gnss_velocity_n"] if k in row}
    return ts, meta


def match_and_compare(egms_ts, gnss_df, gnss_value_col, tolerance_days=3):
    """merge_asof nearest-epoch matching within +/- tolerance_days, then re-reference
    both series to zero at the first common epoch, and compute stats."""
    e = egms_ts.sort_values("date").reset_index(drop=True)
    g = gnss_df[["date", gnss_value_col]].sort_values("date").reset_index(drop=True)
    g = g.rename(columns={gnss_value_col: "gnss_mm"})

    merged = pd.merge_asof(e, g, on="date", direction="nearest",
                            tolerance=pd.Timedelta(days=tolerance_days))
    merged = merged.rename(columns={"value_mm": "egms_mm"})

    matched = merged.dropna(subset=["gnss_mm"]).reset_index(drop=True)
    if len(matched) == 0:
        raise RuntimeError("No matched epochs within tolerance.")

    # re-reference: subtract each series' own value at the first common epoch
    ref_egms = matched["egms_mm"].iloc[0]
    ref_gnss = matched["gnss_mm"].iloc[0]
    matched = matched.copy()
    matched["egms_mm_ref"] = matched["egms_mm"] - ref_egms
    matched["gnss_mm_ref"] = matched["gnss_mm"] - ref_gnss

    # decimal year for trend fits
    matched["decyear"] = matched["date"].dt.year + (matched["date"].dt.dayofyear - 1) / 365.25

    r, p = pearsonr(matched["egms_mm_ref"], matched["gnss_mm_ref"])
    rmse = np.sqrt(np.mean((matched["egms_mm_ref"] - matched["gnss_mm_ref"]) ** 2))

    lr_egms = linregress(matched["decyear"], matched["egms_mm_ref"])
    lr_gnss = linregress(matched["decyear"], matched["gnss_mm_ref"])

    n_egms_total = len(e)
    n_matched = len(matched)
    span_start, span_end = matched["date"].min(), matched["date"].max()

    stats = {
        "n_egms_epochs": n_egms_total,
        "n_matched_epochs": n_matched,
        "match_fraction": n_matched / n_egms_total,
        "window_start": span_start,
        "window_end": span_end,
        "pearson_r": r,
        "pearson_p": p,
        "rmse_mm": rmse,
        "egms_trend_mm_yr": lr_egms.slope,
        "egms_trend_se": lr_egms.stderr,
        "gnss_trend_mm_yr": lr_gnss.slope,
        "gnss_trend_se": lr_gnss.stderr,
        "trend_diff_mm_yr": lr_egms.slope - lr_gnss.slope,
    }
    return matched, stats


def make_timeseries_plot(ax, matched, station, component_label, gap_days=60):
    ax.grid(True, axis="y", linewidth=0.6, zorder=0)
    ax.scatter(matched["date"], matched["egms_mm_ref"], s=14, color=C_BLUE,
               label="EGMS L3 (InSAR)", zorder=3, alpha=0.85)
    # break the GNSS line at data gaps (e.g. D256's 2022-2023 outage) instead of
    # drawing a straight connector across missing time, which would misleadingly
    # imply continuous coverage
    gnss_line = matched[["date", "gnss_mm_ref"]].sort_values("date").reset_index(drop=True)
    gap = gnss_line["date"].diff() > pd.Timedelta(days=gap_days)
    seg_id = gap.cumsum()
    first_seg = True
    for _, seg in gnss_line.groupby(seg_id):
        ax.plot(seg["date"], seg["gnss_mm_ref"], color=C_ORANGE, linewidth=1.4,
                label="NGL GNSS (EU-fixed)" if first_seg else None, zorder=2, alpha=0.9)
        first_seg = False

    # trend lines
    decyear = matched["decyear"].to_numpy()
    lr_e = linregress(decyear, matched["egms_mm_ref"])
    lr_g = linregress(decyear, matched["gnss_mm_ref"])
    x_fit = np.array([decyear.min(), decyear.max()])
    date_fit = matched["date"].iloc[[0, -1]]
    # rebuild fit x positions properly ordered by date
    order = np.argsort(decyear)
    x_sorted = decyear[order]
    date_sorted = matched["date"].to_numpy()[order]
    x_fit = np.array([x_sorted[0], x_sorted[-1]])
    date_fit = np.array([date_sorted[0], date_sorted[-1]])

    ax.plot(date_fit, lr_e.intercept + lr_e.slope * x_fit, color=C_BLUE,
            linewidth=1.6, linestyle="--", zorder=4,
            label=f"EGMS trend {lr_e.slope:+.2f} mm/yr")
    ax.plot(date_fit, lr_g.intercept + lr_g.slope * x_fit, color=C_ORANGE,
            linewidth=1.6, linestyle="--", zorder=4,
            label=f"GNSS trend {lr_g.slope:+.2f} mm/yr")

    ax.set_title(f"{station}: EGMS {component_label} vs NGL GNSS (re-referenced, common epochs)")
    ax.set_ylabel("Displacement (mm)")
    ax.legend(frameon=False, fontsize=8.5, loc="best")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def make_scatter_plot(ax, matched, station, stats, component_label):
    ax.grid(True, linewidth=0.6, zorder=0)
    ax.scatter(matched["gnss_mm_ref"], matched["egms_mm_ref"], s=18, color=C_BLUE,
               alpha=0.75, zorder=3, edgecolor="none")
    lo = min(matched["gnss_mm_ref"].min(), matched["egms_mm_ref"].min())
    hi = max(matched["gnss_mm_ref"].max(), matched["egms_mm_ref"].max())
    pad = 0.05 * (hi - lo) if hi > lo else 1
    lims = [lo - pad, hi + pad]
    ax.plot(lims, lims, color=C_SECONDARY, linewidth=1.2, linestyle="-", zorder=2, label="1:1")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("GNSS (mm)")
    ax.set_ylabel("EGMS (mm)")
    ax.set_title(f"{station}: matched-epoch {component_label} scatter\nr={stats['pearson_r']:.2f}, RMSE={stats['rmse_mm']:.2f} mm, n={stats['n_matched_epochs']}")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def main():
    results = []

    up_matched_csv = f"{DATA}/egms_matched_up.csv"
    east_matched_csv = f"{DATA}/egms_matched_east.csv"

    eu_files = {"OBE4": f"{DATA}/OBE4.EU.tenv3", "D256": f"{DATA}/D256.EU.tenv3"}
    igs20_files = {"OBE4": f"{DATA}/OBE4.IGS20.tenv3", "D256": f"{DATA}/D256.IGS20.tenv3"}

    gnss_eu = {s: load_tenv3(p) for s, p in eu_files.items()}
    gnss_igs20 = {s: load_tenv3(p) for s, p in igs20_files.items()}

    matched_store = {}  # (station, component) -> matched df

    # ---- vertical (Up) comparison ----
    fig_ts, axes_ts = plt.subplots(2, 1, figsize=(9, 8), sharex=False)
    fig_sc, axes_sc = plt.subplots(1, 2, figsize=(9.5, 4.6))

    for i, station in enumerate(["OBE4", "D256"]):
        egms_ts, meta = load_egms_pixel(up_matched_csv, station)
        matched, stats = match_and_compare(egms_ts, gnss_eu[station], "up_mm")
        matched_store[(station, "up")] = matched
        make_timeseries_plot(axes_ts[i], matched, station, "vertical (Up)")
        make_scatter_plot(axes_sc[i], matched, station, stats, "vertical")

        row = {"station": station, "component": "vertical",
               "egms_pid": meta["pid"], "dist_m": meta["dist_to_station_m"]}
        row.update(stats)
        results.append(row)
        print(f"[{station} vertical] n_matched={stats['n_matched_epochs']}/{stats['n_egms_epochs']} "
              f"r={stats['pearson_r']:.3f} RMSE={stats['rmse_mm']:.2f}mm "
              f"EGMS trend={stats['egms_trend_mm_yr']:.2f} GNSS trend={stats['gnss_trend_mm_yr']:.2f} "
              f"diff={stats['trend_diff_mm_yr']:.2f} dist={meta['dist_to_station_m']:.1f}m")

    fig_ts.tight_layout()
    fig_ts.savefig(f"{FIGS}/timeseries_vertical.png", dpi=160, facecolor=C_SURFACE)
    plt.close(fig_ts)

    fig_sc.tight_layout()
    fig_sc.savefig(f"{FIGS}/scatter_vertical.png", dpi=160, facecolor=C_SURFACE)
    plt.close(fig_sc)

    # ---- east-west comparison (secondary/bonus) ----
    fig_ts2, axes_ts2 = plt.subplots(2, 1, figsize=(9, 8), sharex=False)
    fig_sc2, axes_sc2 = plt.subplots(1, 2, figsize=(9.5, 4.6))

    for i, station in enumerate(["OBE4", "D256"]):
        egms_ts, meta = load_egms_pixel(east_matched_csv, station)
        matched, stats = match_and_compare(egms_ts, gnss_eu[station], "east_mm")
        matched_store[(station, "east")] = matched
        make_timeseries_plot(axes_ts2[i], matched, station, "east-west")
        make_scatter_plot(axes_sc2[i], matched, station, stats, "east-west")

        row = {"station": station, "component": "east-west",
               "egms_pid": meta["pid"], "dist_m": meta["dist_to_station_m"]}
        row.update(stats)
        results.append(row)
        print(f"[{station} east-west] n_matched={stats['n_matched_epochs']}/{stats['n_egms_epochs']} "
              f"r={stats['pearson_r']:.3f} RMSE={stats['rmse_mm']:.2f}mm "
              f"EGMS trend={stats['egms_trend_mm_yr']:.2f} GNSS trend={stats['gnss_trend_mm_yr']:.2f} "
              f"diff={stats['trend_diff_mm_yr']:.2f} dist={meta['dist_to_station_m']:.1f}m")

    fig_ts2.tight_layout()
    fig_ts2.savefig(f"{FIGS}/timeseries_eastwest.png", dpi=160, facecolor=C_SURFACE)
    plt.close(fig_ts2)

    fig_sc2.tight_layout()
    fig_sc2.savefig(f"{FIGS}/scatter_eastwest.png", dpi=160, facecolor=C_SURFACE)
    plt.close(fig_sc2)

    # ---- illustrative plate-motion plot: IGS20 vs EU East, D256 & OBE4 ----
    fig_pm, axes_pm = plt.subplots(1, 2, figsize=(10, 4.3), sharey=False)
    plate_motion_stats = {}
    for i, station in enumerate(["D256", "OBE4"]):
        ax = axes_pm[i]
        df_igs = gnss_igs20[station].copy()
        df_eu = gnss_eu[station].copy()
        df_igs["east_mm_ref"] = df_igs["east_mm"] - df_igs["east_mm"].iloc[0]
        df_eu["east_mm_ref"] = df_eu["east_mm"] - df_eu["east_mm"].iloc[0]

        lr_igs = linregress(df_igs["decyear"], df_igs["east_mm_ref"])
        lr_eu = linregress(df_eu["decyear"], df_eu["east_mm_ref"])
        plate_motion_stats[station] = {
            "igs20_east_trend_mm_yr": lr_igs.slope,
            "eu_east_trend_mm_yr": lr_eu.slope,
        }

        ax.grid(True, linewidth=0.6, zorder=0)
        ax.plot(df_igs["date"], df_igs["east_mm_ref"], color=C_RED, linewidth=1.1,
                label=f"IGS20 (global/ITRF): {lr_igs.slope:+.1f} mm/yr", zorder=3)
        ax.plot(df_eu["date"], df_eu["east_mm_ref"], color=C_BLUE, linewidth=1.1,
                label=f"EU-fixed (Eurasia plate): {lr_eu.slope:+.1f} mm/yr", zorder=2)
        ax.set_title(f"{station}: East position, IGS20 vs EU-fixed frame")
        ax.set_ylabel("East displacement (mm)")
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig_pm.tight_layout()
    fig_pm.savefig(f"{FIGS}/plate_motion_frames.png", dpi=160, facecolor=C_SURFACE)
    plt.close(fig_pm)

    print("\nPlate motion (East component) trend, IGS20 vs EU-fixed:")
    for s, d in plate_motion_stats.items():
        print(f"  {s}: IGS20={d['igs20_east_trend_mm_yr']:.2f} mm/yr, "
              f"EU-fixed={d['eu_east_trend_mm_yr']:.2f} mm/yr, "
              f"removed={d['igs20_east_trend_mm_yr']-d['eu_east_trend_mm_yr']:.2f} mm/yr")

    # save results table
    res_df = pd.DataFrame(results)
    res_df.to_csv(f"{DATA}/comparison_results.csv", index=False)
    print(f"\nSaved results table to {DATA}/comparison_results.csv")

    pm_df = pd.DataFrame(plate_motion_stats).T
    pm_df.to_csv(f"{DATA}/plate_motion_results.csv")
    print(f"Saved plate-motion results to {DATA}/plate_motion_results.csv")

    print("\n=== Full results table ===")
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    main()
