"""
Test example: instead of comparing raw epoch-by-epoch EGMS vs GNSS displacement
(dominated by GNSS's own multi-mm daily noise, see analysis.py / report.md),
fit the same trend+annual-harmonic trajectory model independently to each
series and compare the fitted BASIS COEFFICIENTS (trend, annual amplitude,
annual phase). This is offset-reference-invariant (no need to zero both
series at one arbitrary shared epoch) and each coefficient is estimated from
many points, so it averages down GNSS's daily noise far more than a single
epoch-matched pointwise comparison can.

Reuses the already-verified, already-despiked data/loading code from
analysis.py (same station choice, same EU-fixed reference frame, same
outlier flagging) - this script only adds the basis-fit + comparison layer
on top.

Run: module load uv && uv run python basis_coefficient_validation.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from analysis import (DATA, FIGS, C_SURFACE, C_PRIMARY, C_SECONDARY, C_MUTED,
                       C_GRID, C_BASELINE, C_BLUE, C_ORANGE,
                       load_tenv3, load_egms_pixel)

STATIONS = ["OBE4", "D256"]


def fit_trend_annual(decyear, values, n_harmonics=1):
    """OLS fit: value = offset + trend*(t-t0) + sum_k [c_k cos(2pi k (t-t0)) + s_k sin(2pi k (t-t0))].
    Returns coefficients plus derived annual amplitude/phase (day-of-year of the
    seasonal peak), R^2, and the trend's standard error."""
    decyear = np.asarray(decyear, dtype=float)
    values = np.asarray(values, dtype=float)
    t0 = decyear.mean()
    dt = decyear - t0

    cols = [np.ones_like(dt), dt]
    for k in range(1, n_harmonics + 1):
        cols += [np.cos(2 * np.pi * k * dt), np.sin(2 * np.pi * k * dt)]
    X = np.column_stack(cols)

    coef, _, _, _ = np.linalg.lstsq(X, values, rcond=None)
    fitted = X @ coef
    resid = values - fitted
    n, p = X.shape
    dof = max(n - p, 1)
    sigma2 = np.sum(resid ** 2) / dof
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(XtX_inv) * sigma2, 0, None))

    ss_tot = np.sum((values - values.mean()) ** 2)
    r2 = 1 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else np.nan

    c1, s1 = coef[2], coef[3]
    amp1 = float(np.hypot(c1, s1))
    phase1 = float(np.arctan2(s1, c1))  # value's cos/sin combo peaks where 2*pi*dt == phase1
    peak_year = t0 + phase1 / (2 * np.pi)
    peak_doy = (peak_year % 1.0) * 365.25

    # numerical sanity check of the closed-form phase: brute-force the argmax
    # of the fitted seasonal-only term over one year and compare
    grid = np.linspace(0, 1, 3653)
    seasonal_grid = c1 * np.cos(2 * np.pi * grid) + s1 * np.sin(2 * np.pi * grid)
    brute_peak_doy = grid[np.argmax(seasonal_grid)] * 365.25
    # both are periodic in the same frame (dt=0 <-> t0), so they should match directly
    diff = min(abs(brute_peak_doy - (phase1 / (2 * np.pi) % 1.0) * 365.25),
               365.25 - abs(brute_peak_doy - (phase1 / (2 * np.pi) % 1.0) * 365.25))
    assert diff < 2.0, f"phase formula sanity check failed: {diff} days off"

    return {
        "t0": t0, "n_obs": n, "offset_mm": float(coef[0]),
        "trend_mm_yr": float(coef[1]), "trend_se": float(se[1]),
        "annual_amp_mm": amp1, "annual_amp_se": float(np.hypot(se[2], se[3])),
        "annual_peak_doy": peak_doy, "r2": float(r2),
        "coef": coef, "X_cols": cols,
    }


def eval_fit(fit, decyear):
    decyear = np.asarray(decyear, dtype=float)
    dt = decyear - fit["t0"]
    coef = fit["coef"]
    y = coef[0] + coef[1] * dt
    n_harm = (len(coef) - 2) // 2
    for k in range(1, n_harm + 1):
        y = y + coef[2 + 2 * (k - 1)] * np.cos(2 * np.pi * k * dt) + coef[3 + 2 * (k - 1)] * np.sin(2 * np.pi * k * dt)
    return y


def doy_to_month_label(doy):
    d = pd.Timestamp("2021-01-01") + pd.Timedelta(days=doy)
    return d.strftime("%b %d")


def main():
    plt.rcParams.update({
        "figure.facecolor": C_SURFACE, "axes.facecolor": C_SURFACE,
        "axes.edgecolor": C_BASELINE, "axes.labelcolor": C_PRIMARY,
        "text.color": C_PRIMARY, "xtick.color": C_MUTED, "ytick.color": C_MUTED,
        "grid.color": C_GRID, "font.family": "sans-serif", "font.size": 10.5,
        "axes.titlesize": 12, "axes.titleweight": "bold",
    })

    gnss_eu = {s: load_tenv3(f"{DATA}/{s}.EU.tenv3") for s in STATIONS}

    rows = []
    fits = {}  # (station, method) -> fit dict, plus raw (decyear, values) for plotting
    egms_meta = {}

    for station in STATIONS:
        egms_ts, meta = load_egms_pixel(f"{DATA}/egms_matched_up.csv", station)
        egms_meta[station] = meta
        e_decyear = (egms_ts["date"].dt.year + (egms_ts["date"].dt.dayofyear - 1) / 365.25).to_numpy()
        e_fit = fit_trend_annual(e_decyear, egms_ts["value_mm"].to_numpy())
        fits[(station, "EGMS")] = (e_fit, e_decyear, egms_ts["value_mm"].to_numpy())

        g = gnss_eu[station]
        gmask = (g["decyear"] >= e_decyear.min()) & (g["decyear"] <= e_decyear.max())
        g_win = g.loc[gmask]
        g_fit = fit_trend_annual(g_win["decyear"].to_numpy(), g_win["up_mm"].to_numpy())
        fits[(station, "GNSS")] = (g_fit, g_win["decyear"].to_numpy(), g_win["up_mm"].to_numpy())

        for method, fit in [("EGMS", e_fit), ("GNSS", g_fit)]:
            rows.append({
                "station": station, "method": method, "n_obs": fit["n_obs"],
                "trend_mm_yr": fit["trend_mm_yr"], "trend_se": fit["trend_se"],
                "annual_amp_mm": fit["annual_amp_mm"], "annual_amp_se": fit["annual_amp_se"],
                "annual_peak_doy": fit["annual_peak_doy"], "r2": fit["r2"],
            })

        rows.append({
            "station": station, "method": "EGMS_self_reported",
            "n_obs": np.nan,
            "trend_mm_yr": float(meta["mean_velocity"]), "trend_se": np.nan,
            "annual_amp_mm": np.nan, "annual_amp_se": np.nan,
            "annual_peak_doy": np.nan, "r2": np.nan,
        })

    coef_df = pd.DataFrame(rows)
    coef_df.to_csv(f"{DATA}/basis_coefficients.csv", index=False)

    # ---- comparison table: EGMS-fit vs GNSS-fit per station ----
    comp_rows = []
    for station in STATIONS:
        e_fit = fits[(station, "EGMS")][0]
        g_fit = fits[(station, "GNSS")][0]
        phase_diff = e_fit["annual_peak_doy"] - g_fit["annual_peak_doy"]
        phase_diff = (phase_diff + 182.625) % 365.25 - 182.625  # wrap to [-182.6, 182.6]
        comp_rows.append({
            "station": station,
            "trend_diff_mm_yr": e_fit["trend_mm_yr"] - g_fit["trend_mm_yr"],
            "amp_egms_mm": e_fit["annual_amp_mm"], "amp_gnss_mm": g_fit["annual_amp_mm"],
            "amp_ratio_gnss_over_egms": g_fit["annual_amp_mm"] / e_fit["annual_amp_mm"] if e_fit["annual_amp_mm"] > 0 else np.nan,
            "phase_diff_days": phase_diff,
            "egms_gnss_velocity_u_internal_mm_yr": float(egms_meta[station]["gnss_velocity_u"]),
            "external_ngl_gnss_trend_mm_yr": g_fit["trend_mm_yr"],
        })
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(f"{DATA}/basis_coefficient_comparison.csv", index=False)

    print("=== Basis coefficients (trend + annual harmonic), per station/method ===")
    print(coef_df.to_string(index=False))
    print("\n=== EGMS-fit vs GNSS-fit comparison ===")
    print(comp_df.to_string(index=False))

    # ---- Figure A: raw series + fitted trend+annual curve overlay ----
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8), sharex=False)
    for i, station in enumerate(STATIONS):
        ax = axes[i]
        ax.grid(True, axis="y", linewidth=0.6, zorder=0)

        e_fit, e_x, e_y = fits[(station, "EGMS")]
        g_fit, g_x, g_y = fits[(station, "GNSS")]

        g_dates = pd.Timestamp("2020-01-01") + pd.to_timedelta((g_x - 2020.0) * 365.25, unit="D")
        e_dates = pd.Timestamp("2020-01-01") + pd.to_timedelta((e_x - 2020.0) * 365.25, unit="D")

        # break the raw-GNSS line at data gaps (D256's 2022-2023 outage) rather than
        # drawing a straight connector across missing time - same convention as analysis.py
        raw_g = pd.DataFrame({"date": g_dates, "y": g_y - g_fit["offset_mm"]}).sort_values("date")
        gap = raw_g["date"].diff() > pd.Timedelta(days=60)
        for _, seg in raw_g.groupby(gap.cumsum()):
            ax.plot(seg["date"], seg["y"], color=C_ORANGE, linewidth=0.5, alpha=0.35, zorder=1)
        ax.scatter(e_dates, e_y - e_fit["offset_mm"], s=12, color=C_BLUE, alpha=0.6, zorder=3, label="EGMS L3 (raw)")

        # the fitted curves are drawn over the full window even where GNSS has no data
        # (e.g. D256's 2022-2023 gap) - they are a fit, not a claim of observation there
        x_grid = np.linspace(e_x.min(), e_x.max(), 400)
        d_grid = pd.Timestamp("2020-01-01") + pd.to_timedelta((x_grid - 2020.0) * 365.25, unit="D")
        ax.plot(d_grid, eval_fit(e_fit, x_grid) - e_fit["offset_mm"], color=C_BLUE, linewidth=2.0, zorder=4,
                label=f"EGMS fit: {e_fit['trend_mm_yr']:+.2f} mm/yr, amp {e_fit['annual_amp_mm']:.1f} mm, R2={e_fit['r2']:.2f}")
        ax.plot(d_grid, eval_fit(g_fit, x_grid) - g_fit["offset_mm"], color=C_ORANGE, linewidth=2.0, zorder=4, linestyle="--",
                label=f"GNSS fit: {g_fit['trend_mm_yr']:+.2f} mm/yr, amp {g_fit['annual_amp_mm']:.1f} mm, R2={g_fit['r2']:.2f}")

        ax.set_title(f"{station}: trend + annual-harmonic basis fit (offset-invariant, all available epochs)")
        ax.set_ylabel("Displacement, offset-removed (mm)")
        ax.legend(frameon=False, fontsize=8, loc="best")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(f"{FIGS}/basis_fit_overlay.png", dpi=160, facecolor=C_SURFACE)
    plt.close(fig)

    # ---- Figure B: annual-cycle phasor plot (amplitude + peak day-of-year) ----
    fig2 = plt.figure(figsize=(6.2, 6.2))
    ax2 = fig2.add_subplot(111, projection="polar")
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)
    month_starts = pd.date_range("2021-01-01", periods=12, freq="MS").dayofyear
    ax2.set_xticks(2 * np.pi * (month_starts - 1) / 365.25)
    ax2.set_xticklabels([pd.Timestamp(f"2021-{m:02d}-01").strftime("%b") for m in range(1, 13)], fontsize=8.5)
    ax2.set_facecolor(C_SURFACE)
    ax2.grid(color=C_GRID)

    markers = {"OBE4": "o", "D256": "s"}
    colors = {"EGMS": C_BLUE, "GNSS": C_ORANGE}
    for station in STATIONS:
        for method in ["EGMS", "GNSS"]:
            fit = fits[(station, method)][0]
            theta = 2 * np.pi * fit["annual_peak_doy"] / 365.25
            r = fit["annual_amp_mm"]
            ax2.plot([0, theta], [0, r], color=colors[method], linewidth=1.0, alpha=0.5, zorder=2)
            ax2.scatter([theta], [r], s=90, color=colors[method], marker=markers[station], zorder=3,
                        edgecolor="white", linewidth=0.6,
                        label=f"{station} {method}")
    ax2.legend(frameon=False, fontsize=8.5, loc="upper right", bbox_to_anchor=(1.35, 1.1))
    ax2.set_title("Annual-cycle phasor: amplitude (radius, mm) and\npeak day-of-year (angle) - EGMS vs GNSS", pad=28)
    fig2.tight_layout()
    fig2.savefig(f"{FIGS}/annual_phasor.png", dpi=160, facecolor=C_SURFACE)
    plt.close(fig2)

    print(f"\nSaved: {DATA}/basis_coefficients.csv, {DATA}/basis_coefficient_comparison.csv")
    print(f"Saved: {FIGS}/basis_fit_overlay.png, {FIGS}/annual_phasor.png")


if __name__ == "__main__":
    main()
