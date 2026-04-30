from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PECAN_REGION_FOLDERS = {
    "austin": "1minute_data_austin",
    "california": "1minute_data_california",
    "newyork": "1minute_data_newyork",
}

INDIA_CITY_FOLDERS = {
    "bangalore": "bangalore",
    "bengaluru": "bangalore",
    "chennai": "chennai",
    "kochi": "kochi",
    "hyderabad": "hyderabad",
    "delhi": "delhi",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Pecan Street 1-minute data into simulator-ready India profiles "
            "and optionally replace PV with NSRDB city solar."
        )
    )
    parser.add_argument("--data-root", default="data", help="Data root directory")
    parser.add_argument(
        "--city",
        required=True,
        choices=sorted(INDIA_CITY_FOLDERS.keys()),
        help="Target India city for NSRDB solar replacement",
    )
    parser.add_argument("--year", type=int, required=True, help="Target simulation year")
    parser.add_argument(
        "--source-regions",
        default="austin,california,newyork",
        help="Comma-separated Pecan source regions",
    )
    parser.add_argument(
        "--target-kwh-per-day",
        type=float,
        default=6.5,
        help="Target mean daily household load after scaling",
    )
    parser.add_argument(
        "--max-homes",
        type=int,
        default=150,
        help="Maximum homes to keep after wiring (deterministic sample)",
    )
    parser.add_argument(
        "--replace-solar-with-nsrdb",
        action="store_true",
        help="Replace raw Pecan solar with NSRDB city profile",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/pecan_india",
        help="Directory where wired outputs are stored",
    )
    return parser.parse_args()


def _available_usecols(csv_path: Path) -> list[str]:
    cols = pd.read_csv(csv_path, nrows=0).columns
    wanted = ["dataid", "localminute", "grid", "solar", "solar2", "car1", "car2", "battery1"]
    return [col for col in wanted if col in cols]


def _normalize_chunk(chunk: pd.DataFrame, region: str, year: int) -> pd.DataFrame:
    out = chunk.copy()
    out["home_id"] = pd.to_numeric(out.get("dataid"), errors="coerce").astype("Int64")
    ts = pd.to_datetime(out.get("localminute"), errors="coerce", utc=True)
    out["timestamp_ist"] = ts.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

    for col in ["grid", "solar", "solar2", "car1", "car2", "battery1"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out = out[out["timestamp_ist"].dt.year == year]
    if out.empty:
        return out

    out["pv_kw_raw"] = (out["solar"] + out["solar2"]).clip(lower=0.0)
    out["grid_kw"] = out["grid"]
    out["load_kw_raw"] = (out["grid_kw"] + out["pv_kw_raw"]).clip(lower=0.0)
    out["ev_kw"] = (out["car1"] + out["car2"]).clip(lower=0.0)
    out["battery_kw"] = out["battery1"]
    out["source_region"] = region

    return out[
        [
            "home_id",
            "timestamp_ist",
            "source_region",
            "grid_kw",
            "load_kw_raw",
            "pv_kw_raw",
            "ev_kw",
            "battery_kw",
        ]
    ]


def load_pecan_year(data_root: Path, regions: list[str], year: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for region in regions:
        folder = PECAN_REGION_FOLDERS.get(region)
        if folder is None:
            raise ValueError(f"Unsupported region: {region}")

        csv_path = data_root / "Pecan_street_dataport" / folder / f"{folder}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing Pecan CSV: {csv_path}")

        usecols = _available_usecols(csv_path)
        for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=500_000, low_memory=False):
            prepared = _normalize_chunk(chunk, region=region, year=year)
            if not prepared.empty:
                frames.append(prepared)

    if not frames:
        raise RuntimeError("No Pecan rows found for requested year and regions")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["home_id", "timestamp_ist"]).copy()
    combined["home_id"] = combined["home_id"].astype("int64")
    return combined


def scale_household_load(frame: pd.DataFrame, target_kwh_per_day: float) -> pd.DataFrame:
    out = frame.copy()
    daily_kwh = (
        out.assign(date=out["timestamp_ist"].dt.date, step_kwh=out["load_kw_raw"] / 60.0)
        .groupby(["home_id", "date"], as_index=False)["step_kwh"]
        .sum()
    )
    mean_daily = daily_kwh.groupby("home_id", as_index=True)["step_kwh"].mean()

    scale = (target_kwh_per_day / mean_daily).clip(lower=0.3, upper=3.0)
    out = out.join(scale.rename("load_scale"), on="home_id")
    out["load_scale"] = out["load_scale"].fillna(1.0)
    out["load_kw"] = (out["load_kw_raw"] * out["load_scale"]).clip(lower=0.0)
    return out


def load_nsrdb_ghi(data_root: Path, city: str, year: int) -> pd.DataFrame:
    city_folder = INDIA_CITY_FOLDERS[city]
    csv_path = data_root / "nsrdb_himawari" / city_folder / f"{city_folder}_{year}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing NSRDB CSV: {csv_path}")

    nsrdb = pd.read_csv(csv_path, skiprows=2, low_memory=False)
    required = {"Year", "Month", "Day", "Hour", "Minute", "GHI"}
    if not required.issubset(nsrdb.columns):
        raise RuntimeError(f"NSRDB schema missing required columns in {csv_path}")

    ts = pd.to_datetime(
        {
            "year": pd.to_numeric(nsrdb["Year"], errors="coerce"),
            "month": pd.to_numeric(nsrdb["Month"], errors="coerce"),
            "day": pd.to_numeric(nsrdb["Day"], errors="coerce"),
            "hour": pd.to_numeric(nsrdb["Hour"], errors="coerce"),
            "minute": pd.to_numeric(nsrdb["Minute"], errors="coerce"),
        },
        errors="coerce",
    )

    out = pd.DataFrame(
        {
            "timestamp_ist": ts,
            "ghi_wm2": pd.to_numeric(nsrdb["GHI"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["timestamp_ist", "ghi_wm2"]).sort_values("timestamp_ist")
    out = out.drop_duplicates(subset=["timestamp_ist"])
    out = out.set_index("timestamp_ist").resample("1min").interpolate(method="time")
    out["ghi_wm2"] = out["ghi_wm2"].clip(lower=0.0)
    out["ghi_norm"] = (out["ghi_wm2"] / 1000.0).clip(lower=0.0)
    return out.reset_index()


def replace_pv_with_nsrdb(frame: pd.DataFrame, ghi_frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out = out.merge(ghi_frame[["timestamp_ist", "ghi_norm"]], on="timestamp_ist", how="left")
    out["ghi_norm"] = out["ghi_norm"].fillna(0.0)

    pv_peak = out.groupby("home_id", as_index=True)["pv_kw_raw"].quantile(0.98).clip(lower=0.2)
    out = out.join(pv_peak.rename("pv_peak_kw"), on="home_id")
    out["pv_kw"] = (out["pv_peak_kw"] * out["ghi_norm"]).clip(lower=0.0)
    return out


def deterministic_home_sample(frame: pd.DataFrame, max_homes: int) -> pd.DataFrame:
    if max_homes <= 0:
        return frame

    home_ids = sorted(frame["home_id"].dropna().unique().tolist())
    if len(home_ids) <= max_homes:
        return frame

    selected = pd.Series(home_ids).sample(n=max_homes, random_state=42).sort_values().tolist()
    return frame[frame["home_id"].isin(selected)].copy()


def save_outputs(frame: pd.DataFrame, output_dir: Path, city: str, year: int) -> None:
    destination = output_dir / city / str(year)
    destination.mkdir(parents=True, exist_ok=True)

    final = frame[
        [
            "timestamp_ist",
            "home_id",
            "source_region",
            "load_kw",
            "pv_kw",
            "ev_kw",
            "battery_kw",
            "grid_kw",
        ]
    ].copy()
    final = final.sort_values(["home_id", "timestamp_ist"]).reset_index(drop=True)
    final["target_city"] = city

    csv_path = destination / f"pecan_wired_{city}_{year}.csv"
    parquet_path = destination / f"pecan_wired_{city}_{year}.parquet"
    summary_path = destination / f"pecan_wired_{city}_{year}_summary.csv"

    final.to_csv(csv_path, index=False)
    final.to_parquet(parquet_path, index=False)

    summary = (
        final.assign(date=final["timestamp_ist"].dt.date, step_kwh=final["load_kw"] / 60.0)
        .groupby("home_id", as_index=False)
        .agg(
            days=("date", "nunique"),
            mean_daily_kwh=("step_kwh", lambda x: x.sum() / max(1, x.index.size / 1440.0)),
            max_load_kw=("load_kw", "max"),
            max_pv_kw=("pv_kw", "max"),
        )
    )
    summary.to_csv(summary_path, index=False)

    print(f"Wired output CSV: {csv_path}")
    print(f"Wired output Parquet: {parquet_path}")
    print(f"Wired summary: {summary_path}")
    print(f"Rows: {len(final):,}; homes: {final['home_id'].nunique():,}")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    regions = [item.strip().lower() for item in args.source_regions.split(",") if item.strip()]
    if not regions:
        raise RuntimeError("At least one source region is required")

    wired = load_pecan_year(data_root=data_root, regions=regions, year=args.year)
    wired = deterministic_home_sample(wired, max_homes=args.max_homes)
    wired = scale_household_load(wired, target_kwh_per_day=args.target_kwh_per_day)

    if args.replace_solar_with_nsrdb:
        ghi = load_nsrdb_ghi(data_root=data_root, city=args.city, year=args.year)
        wired = replace_pv_with_nsrdb(wired, ghi_frame=ghi)
    else:
        wired["pv_kw"] = wired["pv_kw_raw"]

    save_outputs(
        frame=wired,
        output_dir=output_dir,
        city=INDIA_CITY_FOLDERS[args.city],
        year=args.year,
    )


if __name__ == "__main__":
    main()
