#######################################
# s03_classify_time_series

##############################################################################
#============================================================================#
# 1) Import Statements
#============================================================================#
##############################################################################

import argparse
import s01_download_data as s01
import s02_preprocessing as s02
import numpy as np
import polars as pl
import json
import re

from pathlib import Path
from datetime import date, datetime, timedelta


################################################
# 1.1) Constants
################################################






##############################################################################
#============================================================================#
# 2) Helper Functions
#============================================================================#
##############################################################################




##############################################################################
#============================================================================#
# 3) Processing
#============================================================================#
##############################################################################







##############################################################################
#============================================================================#
# 4) Main
#============================================================================#
##############################################################################


def main():
    pass



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # trend: |velocity| (mm/yr) of the line + annual sine fit
    parser.add_argument("--small-trend", dest="small_trend", type=float, default=2.0,
                        help="mm/yr, |velocity| below this = no trend")
    parser.add_argument("--large-trend", dest="large_trend", type=float, default=10.0,
                        help="mm/yr, |velocity| from this on = large trend")

    # seasonal: the annual sine is large enough and explains enough of the detrended variation
    parser.add_argument("--season-amp-min", dest="season_amp_min", type=float, default=2.0,
                        help="mm, seasonal amplitude from this on = seasonal")
    parser.add_argument("--season-share-min", dest="season_share_min", type=float, default=0.3,
                        help="share of the detrended variation the annual sine explains from this on = seasonal")

    # linear: small rmse and high R² of the linear fit
    parser.add_argument("--lin-rmse-max", dest="lin_rmse_max", type=float, default=3.0,
                        help="mm, rmse of the linear fit up to this = linear")
    parser.add_argument("--lin-r2-min", dest="lin_r2_min", type=float, default=0.8,
                        help="R² of the linear fit from this on = linear (not used for linear_no_trend)")
    parser.add_argument("--stable-amp-max", dest="stable_amp_max", type=float, default=2.0,
                        help="mm, seasonal amplitude up to this = no oscillation (only for linear_no_trend)")

    # noisy: weak signal compared with the noise
    parser.add_argument("--noise-snr", dest="noise_snr", type=float, default=1.0,
                        help="signal / noise below this = noisy")

    # persistent step at the event date of a burst
    parser.add_argument("--window", dest="window", type=int, default=5,
                        help="observations averaged before / after a step position")
    parser.add_argument("--step-z", dest="step_z", type=float, default=10.0,
                        help="strong step (level difference / typical one), only used to find the event date")
    parser.add_argument("--step-event-share", dest="step_event_share", type=float, default=0.2,
                        help="share of the burst's strong-step series on the event date from this on = event; 0 = off")
    parser.add_argument("--event-step-z", dest="event_step_z", type=float, default=3.5,
                        help="level difference at the event date / typical one from this on = step")
    parser.add_argument("--min-step-mm", dest="min_step_mm", type=float, default=10.0,
                        help="mm, smallest step (level difference at the event date) that counts")
    parser.add_argument("--persist-ratio", dest="persist_ratio", type=float, default=0.5,
                        help="the level 5-15 values after the step keeps at least this share of the step")

    # winter gap
    parser.add_argument("--gap-days", dest="gap_days", type=int, default=30,
                        help="this many days or more without data between November and April = winter gap")

    args = parser.parse_args()
