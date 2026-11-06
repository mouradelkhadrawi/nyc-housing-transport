"""
Data pipeline: DOB NOW Build job filings (housing)  x  Motor Vehicle Collisions (transport).

Run `python pipeline.py` to download, clean, aggregate and integrate the two datasets.
Output: data/integrated_zip_month.csv  (consumed by app.py)  and  data/cleaning_log.json

The notebook (nyc_housing_transport.ipynb) is generated from the `# %% [name]` cells below,
so the code you read in the notebook is exactly the code that produced the website's data.
"""

# %% [config]
import json
import re
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# --- Dataset identifiers (NYC Open Data) -------------------------------------------------
HOUSING_ID = "w9ak-ipjd"   # H5  DOB NOW: Build - Job Application Filings   (catalog dataset)
CRASH_ID = "h9gi-nx95"     # T4-style crash data: Motor Vehicle Collisions - Crashes (see feasibility check)

# --- Study window -------------------------------------------------------------------------
# DOB NOW: Build only exists from ~2020 onward, so we start in 2020. The end date is the last
# fully-reported month (the most recent weeks are still being back-filled by the agencies).
START_DATE = "2020-01-01"
END_DATE = "2026-09-30"

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
INTEGRATED_PATH = DATA_DIR / "integrated_zip_month.csv"
LOG_PATH = DATA_DIR / "cleaning_log.json"

# Rough NYC bounding box, used as a domain rule for coordinates
NYC_LAT = (40.45, 40.95)
NYC_LON = (-74.30, -73.65)
BOROUGHS = ["BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"]


# Column-name candidates: API names can differ slightly from the downloadable CSV, so each
# standard name lists the spellings we accept (resolved by pick_column below).
HOUSING_COLUMN_CANDIDATES = {
    "job_id":         ["job_filing_number", "job_number"],
    "filed_date":     ["filing_date", "filed_date", "created_date"],
    "borough":        ["borough"],
    "zip":            ["zip", "zip_code", "zipcode", "postcode"],
    "job_type_raw":   ["job_type"],
    "status":         ["filing_status", "job_status"],
    "proposed_units": ["proposed_dwelling_units", "proposed_no_of_dwelling_units", "proposed_units"],
    "existing_units": ["existing_dwelling_units", "existing_no_of_dwelling_units"],
    "cost":           ["initial_cost", "estimated_job_costs", "total_construction_floor_area"],
    "lat":            ["latitude", "gis_latitude"],
    "lon":            ["longitude", "gis_longitude"],
}

CRASH_COLUMN_CANDIDATES = {
    "crash_id":     ["collision_id"],
    "crash_date":   ["crash_date"],
    "borough":      ["borough"],
    "zip":          ["zip_code", "zip"],
    "lat":          ["latitude"],
    "lon":          ["longitude"],
    "injured":      ["number_of_persons_injured"],
    "killed":       ["number_of_persons_killed"],
    "ped_injured":  ["number_of_pedestrians_injured"],
    "cyc_injured":  ["number_of_cyclist_injured", "number_of_cyclists_injured"],
    "mot_injured":  ["number_of_motorist_injured", "number_of_motorists_injured"],
    "factor":       ["contributing_factor_vehicle_1"],
}
CRASH_COUNT_COLUMNS = ["injured", "killed", "ped_injured", "cyc_injured", "mot_injured"]


# %% [helpers]
def pick_column(df, candidates, label, required=True):
    """Return the first candidate column present in df (case/space-insensitive)."""
    normalised = {re.sub(r"[\s]+", "_", c.strip().lower()): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand in normalised:
            return normalised[cand]
    if required:
        raise KeyError(f"Could not find a column for '{label}'. Tried {candidates}. "
                       f"Available: {list(df.columns)}")
    return None


def standardize_zip(series):
    """ZIPs -> 5-digit strings. Anything that is not a plausible NYC ZIP becomes NaN.

    Real data contains values like '11201-1234', '11201.0', 'N/A', '0', '10000'.
    The 10001-11697 window is a coarse domain rule; the later ZIP-universe intersection
    between the two datasets removes the remaining non-NYC ZIPs.
    """
    five = series.astype(str).str.extract(r"(\d{5})")[0]
    numeric = pd.to_numeric(five, errors="coerce")
    return five.where(numeric.between(10001, 11697))


def standardize_borough(series):
    """Upper-case, trim and map spelling variants / numeric DOB codes to the 5 borough names."""
    cleaned = series.astype(str).str.upper().str.strip()
    mapping = {
        "THE BRONX": "BRONX", "BX": "BRONX", "2": "BRONX",
        "BK": "BROOKLYN", "BKLYN": "BROOKLYN", "3": "BROOKLYN",
        "NY": "MANHATTAN", "NEW YORK": "MANHATTAN", "MN": "MANHATTAN", "1": "MANHATTAN",
        "QN": "QUEENS", "4": "QUEENS",
        "SI": "STATEN ISLAND", "STATEN IS": "STATEN ISLAND", "STATEN ISLAND ": "STATEN ISLAND",
        "5": "STATEN ISLAND",
    }
    cleaned = cleaned.replace(mapping)
    return cleaned.where(cleaned.isin(BOROUGHS))


def nullify_outside_nyc(df, lat_col="lat", lon_col="lon"):
    """Coordinates outside the NYC bounding box (incl. the infamous (0, 0)) -> NaN."""
    df = df.copy()
    bad = ~(df[lat_col].between(*NYC_LAT) & df[lon_col].between(*NYC_LON))
    df.loc[bad, [lat_col, lon_col]] = np.nan
    return df, int(bad.sum())


def impute_zip_from_neighbors(df, zip_col="zip", lat_col="lat", lon_col="lon", max_dist_m=300):
    """Fill a missing ZIP with the ZIP of the nearest record (same dataset) that has one.

    Many crash rows have coordinates but no ZIP. Dropping them would bias the data toward
    car-heavy ZIPs with better reporting, so we borrow the ZIP of the nearest located record.
    The 300 m cutoff stops us from borrowing across a large gap (e.g. a park or a river).
    Returns (new_df, number_imputed).
    """
    df = df.copy()
    known = df[df[zip_col].notna() & df[lat_col].notna()]
    todo = df[df[zip_col].isna() & df[lat_col].notna()]
    if known.empty or todo.empty:
        return df, 0

    lat0 = 40.7  # project degrees to approximate metres around NYC's latitude

    def to_xy(frame):
        return np.column_stack([
            frame[lon_col].to_numpy() * 111_320 * np.cos(np.radians(lat0)),
            frame[lat_col].to_numpy() * 110_540,
        ])

    tree = cKDTree(to_xy(known))
    dist, idx = tree.query(to_xy(todo), k=1, distance_upper_bound=max_dist_m)
    found = np.isfinite(dist)
    df.loc[todo.index[found], zip_col] = known[zip_col].to_numpy()[idx[found]]
    return df, int(found.sum())


def iqr_bounds(series, k=1.5):
    """Tukey fences. Used to FLAG (not delete) statistical outliers."""
    q1, q3 = series.quantile([0.25, 0.75])
    spread = q3 - q1
    return q1 - k * spread, q3 + k * spread


def group_job_type(raw_type):
    """Collapse DOB job types into 4 groups: NB (new building), ALT, DM (demolition), OTHER."""
    text = str(raw_type).lower()
    if "alt-co" in text or "alteration co" in text:
        return "ALT"            # 'ALT-CO - New Building with Existing Elements to Remain'
    if "new building" in text:
        return "NB"
    if "demol" in text:
        return "DM"
    if "alteration" in text or text.startswith("alt"):
        return "ALT"
    return "OTHER"


# %% [fetch]
import time
from urllib.error import HTTPError, URLError

API_URL = "https://data.cityofnewyork.us/resource/{}.csv?{}"


def read_csv_with_retry(url, attempts=6):
    """Read a CSV URL, retrying temporary server errors (429 / 5xx) with growing waits.

    NYC Open Data sometimes answers 503 under load or when a request is too heavy; waiting a
    few seconds and trying again usually works. Real errors (400, 404) are raised at once.
    """
    for attempt in range(1, attempts + 1):
        try:
            return pd.read_csv(url, low_memory=False)
        except (HTTPError, URLError) as err:
            code = getattr(err, "code", None)
            retryable = code is None or code == 429 or code >= 500
            if not retryable or attempt == attempts:
                raise
            wait = 5 * 2 ** (attempt - 1)          # 5, 10, 20, 40, 80 seconds
            print(f"  server answered {code or err}; retry {attempt}/{attempts - 1} in {wait}s")
            time.sleep(wait)


def list_columns(dataset_id):
    """Fetch a single row to learn the dataset's real (API) column names."""
    url = API_URL.format(dataset_id, urlencode({"$limit": 1}))
    return list(read_csv_with_retry(url).columns)


def needed_columns(all_columns, candidate_map):
    """Subset of the dataset's columns that our standard fields map to (pick_column logic)."""
    probe = pd.DataFrame(columns=all_columns)
    found = [pick_column(probe, cands, std, required=False) for std, cands in candidate_map.items()]
    return [col for col in dict.fromkeys(found) if col]


def fetch_socrata(dataset_id, where=None, select=None, page_size=20_000, max_rows=3_000_000):
    """Download a Socrata dataset page by page (the API returns only 1,000 rows by default).

    - `select` limits the download to the columns we need (these datasets have 30-100+ columns,
      and asking for all of them is what makes big requests time out).
    - Results are cached in data/raw/<dataset_id>.csv so re-running the notebook is instant
      and does not hammer the API.
    """
    cache_file = RAW_DIR / f"{dataset_id}.csv"
    if cache_file.exists():
        return pd.read_csv(cache_file, low_memory=False)

    frames, offset = [], 0
    while offset < max_rows:
        params = {"$limit": page_size, "$offset": offset, "$order": ":id"}
        if where:
            params["$where"] = where
        if select:
            params["$select"] = ",".join(select)
        chunk = read_csv_with_retry(API_URL.format(dataset_id, urlencode(params)))
        if chunk.empty:
            break
        frames.append(chunk)
        offset += page_size
        print(f"  {dataset_id}: {offset:,} rows requested so far")
        if len(chunk) < page_size:
            break

    df = pd.concat(frames, ignore_index=True)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_file, index=False)
    return df


def load_dataset(dataset_id, candidate_map, date_key):
    """Download one dataset: real column names first, then only the needed columns since START_DATE."""
    cache_file = RAW_DIR / f"{dataset_id}.csv"
    if cache_file.exists():                      # already downloaded: no network needed
        return pd.read_csv(cache_file, low_memory=False)
    all_columns = list_columns(dataset_id)
    probe = pd.DataFrame(columns=all_columns)
    date_field = pick_column(probe, candidate_map[date_key], date_key)
    select = needed_columns(all_columns, candidate_map)
    print(f"{dataset_id}: {len(all_columns)} columns available, downloading {len(select)}: {select}")
    try:
        return fetch_socrata(dataset_id, where=f"{date_field} >= '{START_DATE}T00:00:00'", select=select)
    except HTTPError as err:
        if err.code != 400:
            raise
        print("  server rejected the date filter; downloading without it (filtered locally later)")
        return fetch_socrata(dataset_id, select=select)


def load_housing_raw():
    return load_dataset(HOUSING_ID, HOUSING_COLUMN_CANDIDATES, "filed_date")


def load_crashes_raw():
    return load_dataset(CRASH_ID, CRASH_COLUMN_CANDIDATES, "crash_date")


# %% [clean_housing]
def clean_housing(raw):
    """Pre-integration cleaning of DOB NOW job filings. Returns (clean_df, log)."""
    log = {"rows_raw": len(raw), "columns_raw": raw.shape[1]}

    # 1. Resolve and rename the columns we need (everything else is irrelevant to our questions)
    rename = {}
    for std_name, candidates in HOUSING_COLUMN_CANDIDATES.items():
        found = pick_column(raw, candidates, std_name, required=std_name in
                            ("job_id", "filed_date", "borough", "zip", "job_type_raw"))
        if found:
            rename[found] = std_name
    df = raw[list(rename)].rename(columns=rename).copy()
    for optional in ("status", "proposed_units", "existing_units", "cost", "lat", "lon"):
        if optional not in df.columns:
            df[optional] = np.nan

    # 2. Duplicates: a job has one initial filing (…-I1) plus post-approval amendments (…-P1, -P2).
    #    Counting amendments would inflate "construction activity", so keep one row per job.
    df["job_id"] = df["job_id"].astype(str).str.strip()
    is_amendment = df["job_id"].str.contains(r"-P\d+$", regex=True)
    log["amendment_rows_dropped"] = int(is_amendment.sum())
    df = df[~is_amendment]
    before = len(df)
    df = df.drop_duplicates(subset="job_id", keep="first")
    log["duplicate_job_ids_dropped"] = before - len(df)

    # 3. Types / formats
    df["filed_date"] = pd.to_datetime(df["filed_date"], errors="coerce")
    for col in ("proposed_units", "existing_units", "cost", "lat", "lon"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["borough"] = standardize_borough(df["borough"])
    df["zip"] = standardize_zip(df["zip"])
    df["job_group"] = df["job_type_raw"].map(group_job_type)

    # 4. Dates: missing, outside the study window (incl. typos in the future) -> drop
    in_window = df["filed_date"].between(START_DATE, END_DATE)
    log["rows_dropped_bad_or_out_of_window_date"] = int((~in_window).sum())
    df = df[in_window]

    # 5. Coordinates: outside NYC -> NaN (kept as a row, we only need coords to repair ZIPs/map)
    df, n_bad_coords = nullify_outside_nyc(df)
    log["coords_nullified_outside_nyc"] = n_bad_coords

    # 6. Missing ZIP: impute from nearest located filing, then drop what is still missing
    log["zip_missing_before"] = int(df["zip"].isna().sum())
    df, n_imputed = impute_zip_from_neighbors(df)
    log["zip_imputed_from_neighbors"] = n_imputed
    log["rows_dropped_no_zip"] = int(df["zip"].isna().sum())
    df = df.dropna(subset=["zip"])

    # 7. Outliers. Domain rule for dwelling units: negative is impossible and no NYC building
    #    has > 3,000 units -> NaN. Costs: flag with IQR on log scale (heavily right-skewed),
    #    but keep the rows because a huge tower really is expensive.
    bad_units = (df["proposed_units"] < 0) | (df["proposed_units"] > 3000)
    df.loc[bad_units, "proposed_units"] = np.nan
    log["proposed_units_nullified_domain_rule"] = int(bad_units.sum())
    df.loc[df["existing_units"] < 0, "existing_units"] = np.nan
    positive_cost = df["cost"].where(df["cost"] > 0)
    low, high = iqr_bounds(np.log1p(positive_cost.dropna()))
    df["cost_outlier_flag"] = (np.log1p(positive_cost) > high) | (np.log1p(positive_cost) < low)
    log["cost_outliers_flagged_iqr"] = int(df["cost_outlier_flag"].sum())

    # 8. New dwelling units only make sense for new buildings (alterations may re-state the same units)
    df["new_units"] = np.where(df["job_group"] == "NB", df["proposed_units"].fillna(0), 0.0)

    df = df.reset_index(drop=True)
    log["rows_clean"] = len(df)
    return df, log


# %% [clean_crashes]
def clean_crashes(raw):
    """Pre-integration cleaning of Motor Vehicle Collisions. Returns (clean_df, log)."""
    log = {"rows_raw": len(raw), "columns_raw": raw.shape[1]}

    rename = {}
    for std_name, candidates in CRASH_COLUMN_CANDIDATES.items():
        found = pick_column(raw, candidates, std_name,
                            required=std_name in ("crash_id", "crash_date", "zip"))
        if found:
            rename[found] = std_name
    df = raw[list(rename)].rename(columns=rename).copy()
    for optional in ("borough", "lat", "lon", "factor") + tuple(CRASH_COUNT_COLUMNS):
        if optional not in df.columns:
            df[optional] = np.nan

    # Duplicates
    before = len(df)
    df = df.drop_duplicates(subset="crash_id", keep="first")
    log["duplicate_crash_ids_dropped"] = before - len(df)

    # Types / formats
    df["crash_date"] = pd.to_datetime(df["crash_date"], errors="coerce")
    for col in ["lat", "lon"] + CRASH_COUNT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["borough"] = standardize_borough(df["borough"])
    df["zip"] = standardize_zip(df["zip"])
    df["factor"] = df["factor"].astype(str).str.strip().str.title().replace({"Nan": np.nan, "": np.nan})

    in_window = df["crash_date"].between(START_DATE, END_DATE)
    log["rows_dropped_bad_or_out_of_window_date"] = int((~in_window).sum())
    df = df[in_window]

    df, n_bad_coords = nullify_outside_nyc(df)
    log["coords_nullified_outside_nyc"] = n_bad_coords

    # Missing counts: a blank injury count on a crash record means "none recorded" -> 0.
    # Negative counts are impossible -> 0 as well; implausible counts (> 40) are data-entry errors.
    log["missing_counts_filled_zero"] = int(df[CRASH_COUNT_COLUMNS].isna().sum().sum())
    df[CRASH_COUNT_COLUMNS] = df[CRASH_COUNT_COLUMNS].fillna(0).clip(lower=0)
    absurd = df[CRASH_COUNT_COLUMNS].gt(40).any(axis=1)
    log["rows_dropped_implausible_counts"] = int(absurd.sum())
    df = df[~absurd]

    # Consistency: sub-category injuries cannot exceed total injuries
    inconsistent = (df[["ped_injured", "cyc_injured", "mot_injured"]].sum(axis=1) > df["injured"])
    log["rows_with_inconsistent_injury_breakdown"] = int(inconsistent.sum())
    df.loc[inconsistent, "injured"] = df.loc[inconsistent, ["ped_injured", "cyc_injured",
                                                            "mot_injured"]].sum(axis=1)

    # Missing ZIP: impute from nearest crash with a ZIP, drop the rest
    log["zip_missing_before"] = int(df["zip"].isna().sum())
    df, n_imputed = impute_zip_from_neighbors(df)
    log["zip_imputed_from_neighbors"] = n_imputed
    log["rows_dropped_no_zip"] = int(df["zip"].isna().sum())
    df = df.dropna(subset=["zip"])

    df = df.reset_index(drop=True)
    log["rows_clean"] = len(df)
    return df, log


# %% [aggregate]
def month_start(series):
    """Truncate timestamps to the first day of their month."""
    return series.dt.to_period("M").dt.to_timestamp()


def aggregate_housing(housing):
    """Housing filings -> one row per (ZIP, month): counts by job group + new dwelling units."""
    work = housing.assign(month=month_start(housing["filed_date"]))
    counts = pd.crosstab([work["zip"], work["month"]], work["job_group"])
    for group in ("NB", "ALT", "DM", "OTHER"):
        if group not in counts.columns:
            counts[group] = 0
    counts = counts.rename(columns={"NB": "filings_nb", "ALT": "filings_alt",
                                    "DM": "filings_dm", "OTHER": "filings_other"})
    out = counts[["filings_nb", "filings_alt", "filings_dm", "filings_other"]].copy()
    out["filings_total"] = out.sum(axis=1)
    out["new_units"] = work.groupby(["zip", "month"])["new_units"].sum()
    return out.reset_index()


def aggregate_crashes(crashes):
    """Crashes -> one row per (ZIP, month): crash count and injury/fatality sums."""
    work = crashes.assign(month=month_start(crashes["crash_date"]))
    grouped = work.groupby(["zip", "month"])
    out = grouped[CRASH_COUNT_COLUMNS].sum()
    out["crashes"] = grouped.size()
    return out.reset_index()


def build_zip_reference(housing, crashes):
    """One row per ZIP: modal borough (across both datasets) and centroid for the map.

    Crash records often lack a borough, so we derive it from the ZIP's most common borough.
    """
    both = pd.concat([
        housing[["zip", "borough", "lat", "lon"]],
        crashes[["zip", "borough", "lat", "lon"]],
    ], ignore_index=True)
    borough = (both.dropna(subset=["borough"]).groupby("zip")["borough"]
               .agg(lambda s: s.mode().iat[0]).rename("borough"))
    centroid = both.groupby("zip")[["lat", "lon"]].mean().rename(
        columns={"lat": "zip_lat", "lon": "zip_lon"})
    return centroid.join(borough).reset_index()


# %% [integrate]
MIN_FILINGS_PER_ZIP = 10   # ZIPs with almost no filings / crashes (airports, PO boxes) are noise
MIN_CRASHES_PER_ZIP = 10


def integrate(housing_agg, crash_agg, zip_reference):
    """Integrate at ZIP x month granularity.

    Why ZIP x month?  - Both datasets have a ZIP and a date, so no row-level matching is needed
                        (a filing is a paperwork event, a crash is a street event; they are never
                        the same record).
                      - ZIP is the finest geography both share reliably; month is fine enough to
                        see seasonality and lags, coarse enough that counts are not all zero.
    Join logic        - ZIP universe = ZIPs present in BOTH datasets (inner on ZIP), so we compare
                        like with like.
                      - Full ZIP x month grid, then LEFT joins: a missing ZIP-month in a source
                        means "no events", i.e. a true zero (resolved in post-integration cleaning).
    Returns (integrated_df_with_NaNs_from_join, log).
    """
    zips = sorted(set(housing_agg["zip"]) & set(crash_agg["zip"]))
    months = pd.date_range(START_DATE, END_DATE, freq="MS")
    grid = pd.MultiIndex.from_product([zips, months], names=["zip", "month"]).to_frame(index=False)

    merged = (grid.merge(housing_agg, on=["zip", "month"], how="left")
                  .merge(crash_agg, on=["zip", "month"], how="left")
                  .merge(zip_reference, on="zip", how="left"))

    log = {
        "zips_housing": int(housing_agg["zip"].nunique()),
        "zips_crashes": int(crash_agg["zip"].nunique()),
        "zips_in_both": len(zips),
        "months": len(months),
        "grid_rows": len(grid),
        "grid_rows_without_housing_record": int(merged["filings_total"].isna().sum()),
        "grid_rows_without_crash_record": int(merged["crashes"].isna().sum()),
    }
    return merged, log


# %% [post_clean]
HOUSING_METRICS = ["filings_nb", "filings_alt", "filings_dm", "filings_other", "filings_total", "new_units"]
CRASH_METRICS = ["crashes", "injured", "killed", "ped_injured", "cyc_injured", "mot_injured"]


def post_integration_clean(merged):
    """Resolve what the join introduced. Returns (clean_df, log)."""
    df = merged.copy()
    log = {"rows_before": len(df)}

    # 1. NaNs from the join: absence of a record in a ZIP-month = zero events (true zeros).
    count_cols = HOUSING_METRICS + CRASH_METRICS
    log["join_nans_filled_with_zero"] = int(df[count_cols].isna().sum().sum())
    df[count_cols] = df[count_cols].fillna(0)

    # 2. Data types: counts were floats after the left join; make them integers (units stay numeric).
    for col in [c for c in count_cols if c != "new_units"]:
        df[col] = df[col].astype(int)
    df["new_units"] = df["new_units"].round().astype(int)
    df["zip"] = df["zip"].astype(str)

    # 3. Redundant / inconsistent columns: total filings must equal the sum of its parts
    parts = df[["filings_nb", "filings_alt", "filings_dm", "filings_other"]].sum(axis=1)
    log["filings_total_mismatch_rows"] = int((parts != df["filings_total"]).sum())
    df["filings_total"] = parts
    # injured must cover its breakdown (cleaned earlier, re-check after aggregation)
    breakdown = df[["ped_injured", "cyc_injured", "mot_injured"]].sum(axis=1)
    log["injured_breakdown_mismatch_rows"] = int((breakdown > df["injured"]).sum())
    df["injured"] = np.maximum(df["injured"], breakdown)

    # 4. Drop ZIPs with too little data to be interpretable
    totals = df.groupby("zip")[["filings_total", "crashes"]].sum()
    keep = totals[(totals["filings_total"] >= MIN_FILINGS_PER_ZIP) &
                  (totals["crashes"] >= MIN_CRASHES_PER_ZIP)].index
    log["zips_dropped_too_sparse"] = int(df["zip"].nunique() - len(keep))
    df = df[df["zip"].isin(keep)]

    # 5. ZIPs whose borough or centroid is still missing (should be none) -> drop
    log["rows_dropped_missing_reference"] = int(df[["borough", "zip_lat", "zip_lon"]].isna().any(axis=1).sum())
    df = df.dropna(subset=["borough", "zip_lat", "zip_lon"])

    # 6. Convenience time columns for the website
    df["year"] = df["month"].dt.year
    df["month_num"] = df["month"].dt.month

    df = df.sort_values(["zip", "month"]).reset_index(drop=True)
    assert df.isna().sum().sum() == 0, "Unexpected NaNs remain after post-integration cleaning"
    log.update({"rows_after": len(df), "zips_final": int(df["zip"].nunique()),
                "columns_final": list(df.columns)})
    return df, log


# %% [main]
def run_pipeline():
    """Download -> clean -> aggregate -> integrate -> post-clean -> save."""
    print("Downloading (cached after first run)...")
    housing_raw, crashes_raw = load_housing_raw(), load_crashes_raw()
    housing, housing_log = clean_housing(housing_raw)
    crashes, crash_log = clean_crashes(crashes_raw)
    merged, integration_log = integrate(aggregate_housing(housing), aggregate_crashes(crashes),
                                        build_zip_reference(housing, crashes))
    final, post_log = post_integration_clean(merged)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    final.to_csv(INTEGRATED_PATH, index=False)
    LOG_PATH.write_text(json.dumps({"housing": housing_log, "crashes": crash_log,
                                    "integration": integration_log, "post_integration": post_log},
                                   indent=2, default=str))
    print(f"Saved {len(final):,} rows ({final['zip'].nunique()} ZIPs) to {INTEGRATED_PATH}")
    return final


if __name__ == "__main__":
    run_pipeline()
