"""
Stream through the large EGMS L3 ortho CSVs (vertical + east-west) and find the
nearest pixel to each GNSS station (OBE4, D256), for both components.

Files are ~250MB uncompressed; we read them in chunks straight out of the zip
so we never hold the whole thing in memory. For each chunk we compute the
squared distance (in EPSG:3035 metres) from every pixel to each station and
keep a running best (row + distance) per station. At the end we save the
winning rows (full time series) to small CSVs in data/ for downstream use.
"""
import zipfile
import numpy as np
import pandas as pd
from pyproj import Transformer

UP_ZIP = "/dss/dsshome1/0C/di54haf/test_egms/egms_density_test/data/EGMS_ortho_up.zip"
UP_MEMBER = "EGMS_L3_E44N27_100km_U_2020_2024_1.csv"
EAST_ZIP = "/dss/dsshome1/0C/di54haf/test_egms/egms_density_test/data/EGMS_ortho_east.zip"
EAST_MEMBER = "EGMS_L3_E44N27_100km_E_2020_2024_1.csv"

STATIONS = {
    "OBE4": (48.0848, 11.2779),
    "D256": (48.1411, 11.5901),
}

OUT_DIR = "/dss/dsshome1/0C/di54haf/LOSTInPLACE-Line-Of-Sight-To-INSAR-derived-continuous-DisPLACEment/test/data"

# lat/lon (EPSG:4326) -> EPSG:3035 (EGMS native CRS)
to_3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
station_xy = {}
for name, (lat, lon) in STATIONS.items():
    x, y = to_3035.transform(lon, lat)
    station_xy[name] = (x, y)
    print(f"{name}: lat/lon=({lat},{lon}) -> EPSG:3035 ({x:.1f}, {y:.1f})")


def find_nearest(zip_path, member, station_xy, chunksize=50000):
    """Stream the CSV and return, for each station, the nearest row (as dict) + distance (m)."""
    best = {name: (np.inf, None) for name in station_xy}
    z = zipfile.ZipFile(zip_path)
    with z.open(member) as f:
        reader = pd.read_csv(f, chunksize=chunksize)
        nrows = 0
        for chunk in reader:
            nrows += len(chunk)
            ex = chunk["easting"].to_numpy()
            ny = chunk["northing"].to_numpy()
            for name, (sx, sy) in station_xy.items():
                d2 = (ex - sx) ** 2 + (ny - sy) ** 2
                idx = np.argmin(d2)
                dmin = np.sqrt(d2[idx])
                if dmin < best[name][0]:
                    best[name] = (dmin, chunk.iloc[idx].to_dict())
        print(f"  scanned {nrows} rows from {member}")
    return best


if __name__ == "__main__":
    print("\n--- Vertical (U) component ---")
    best_up = find_nearest(UP_ZIP, UP_MEMBER, station_xy)
    for name, (dist, row) in best_up.items():
        print(f"{name}: nearest UP pixel pid={row['pid']} dist={dist:.1f} m "
              f"(easting={row['easting']}, northing={row['northing']}, mean_velocity={row['mean_velocity']})")

    print("\n--- East-West (E) component ---")
    best_east = find_nearest(EAST_ZIP, EAST_MEMBER, station_xy)
    for name, (dist, row) in best_east.items():
        print(f"{name}: nearest EAST pixel pid={row['pid']} dist={dist:.1f} m "
              f"(easting={row['easting']}, northing={row['northing']}, mean_velocity={row['mean_velocity']})")

    # save matched rows
    up_rows = []
    for name, (dist, row) in best_up.items():
        r = dict(row)
        r["station"] = name
        r["dist_to_station_m"] = dist
        up_rows.append(r)
    pd.DataFrame(up_rows).to_csv(f"{OUT_DIR}/egms_matched_up.csv", index=False)

    east_rows = []
    for name, (dist, row) in best_east.items():
        r = dict(row)
        r["station"] = name
        r["dist_to_station_m"] = dist
        east_rows.append(r)
    pd.DataFrame(east_rows).to_csv(f"{OUT_DIR}/egms_matched_east.csv", index=False)

    print("\nSaved matched pixel rows to data/egms_matched_up.csv and data/egms_matched_east.csv")

    # sanity check: same pid in both up and east files?
    for name in STATIONS:
        pid_up = best_up[name][1]["pid"]
        pid_east = best_east[name][1]["pid"]
        match = "MATCH" if pid_up == pid_east else "DIFFERENT"
        print(f"{name}: up pid={pid_up}  east pid={pid_east}  -> {match}")
