import io
import os
import time
import zipfile

import pandas as pd
import requests


def load_env_file(env_path: str = ".env") -> None:
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_env_file()

API_KEY = os.getenv("NSRDB_API_KEY", "")
EMAIL = os.getenv("NSRDB_EMAIL", "")
BASE_URL = "https://developer.nrel.gov/api/nsrdb/v2/solar/himawari-download.json"

CITIES = {
    "bangalore": ["POINT(77.5946 12.9716)"],
    "chennai": ["POINT(80.2707 13.0827)"],
    
    "kochi": ["POINT(76.2673 9.9312)"],
    "hyderabad": ["POINT(78.4867 17.3850)"],
    "delhi": ["POINT(77.1025 28.7041)"],
}

YEARS = ["2016", "2017", "2018", "2019", "2020"]
ATTRIBUTES = (
    "ghi,dni,dhi,air_temperature,wind_speed,relative_humidity,"
    "cloud_type,solar_zenith_angle"
)
OUTPUT_DIR = "data/nsrdb_himawari"


def get_response_json_and_handle_errors(response: requests.Response) -> dict:
    if response.status_code != 200:
        safe_details = response.reason
        try:
            error_json = response.json()
            if isinstance(error_json, dict):
                if "errors" in error_json and isinstance(error_json["errors"], list):
                    safe_details = "; ".join(str(item) for item in error_json["errors"])
                elif "error" in error_json and isinstance(error_json["error"], dict):
                    safe_details = str(error_json["error"].get("message", response.reason))
        except ValueError:
            pass

        raise RuntimeError(
            f"API request failed ({response.status_code} {response.reason}): {safe_details}"
        )

    try:
        response_json = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"API returned non-JSON response body: {response.text}"
        ) from exc

    errors = response_json.get("errors", [])
    if errors:
        joined_errors = "\n".join(errors)
        raise RuntimeError(f"API returned errors:\n{joined_errors}")

    return response_json


def submit_download_request(city: str, year: str, wkt_point: str) -> str:
    payload = {
        "attributes": ATTRIBUTES,
        "interval": "30",
        "names": [year],
        "wkt": wkt_point,
        "api_key": API_KEY,
        "email": EMAIL,
    }
    headers = {"x-api-key": API_KEY}

    response = requests.post(BASE_URL, data=payload, headers=headers, timeout=60)
    response_json = get_response_json_and_handle_errors(response)
    outputs = response_json.get("outputs", {})

    download_url = outputs.get("downloadUrl")
    if not download_url:
        raise RuntimeError(
            f"No download URL returned for city={city}, year={year}. Response: {response_json}"
        )

    message = outputs.get("message", "Request accepted")
    print(f"[{city} {year}] {message}")
    return download_url


def submit_download_request_with_fallback(city: str, year: str, wkt_points: list[str]) -> str:
    last_error: Exception | None = None
    for index, wkt_point in enumerate(wkt_points, start=1):
        try:
            download_url = submit_download_request(city, year, wkt_point)
            if index > 1:
                print(f"[{city} {year}] fallback coordinate worked: {wkt_point}")
            return download_url
        except RuntimeError as exc:
            message = str(exc)
            if "No data available at the provided location" not in message:
                raise
            print(f"[{city} {year}] no data at {wkt_point}, trying next fallback.")
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"No coordinates configured for city={city}")


def wait_and_download_zip(download_url: str, city: str, year: str) -> bytes:
    for attempt in range(1, 61):
        response = requests.get(download_url, timeout=120)
        if response.status_code == 200:
            return response.content

        if response.status_code in (403, 404, 425, 429, 500, 502, 503, 504):
            print(
                f"[{city} {year}] file not ready yet "
                f"(attempt {attempt}/60, status={response.status_code})."
            )
            time.sleep(10)
            continue

        raise RuntimeError(
            f"Unexpected response while downloading city={city}, year={year}: "
            f"{response.status_code} {response.text[:400]}"
        )

    raise TimeoutError(
        f"Timed out waiting for NSRDB file for city={city}, year={year}."
    )


def content_to_dataframe(content: bytes) -> pd.DataFrame:
    if zipfile.is_zipfile(io.BytesIO(content)):
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            csv_files = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not csv_files:
                raise RuntimeError("ZIP downloaded successfully but contains no CSV files.")
            with archive.open(csv_files[0]) as csv_stream:
                return pd.read_csv(csv_stream, low_memory=False)

    return pd.read_csv(io.BytesIO(content), low_memory=False)


def main() -> None:
    if not API_KEY:
        raise RuntimeError("Missing NSRDB_API_KEY. Set it in .env or the shell environment.")
    if not EMAIL:
        raise RuntimeError("Missing NSRDB_EMAIL. Set it in .env or the shell environment.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    failures: list[dict[str, str]] = []

    for city, wkt_points in CITIES.items():
        city_dir = os.path.join(OUTPUT_DIR, city)
        os.makedirs(city_dir, exist_ok=True)

        for year in YEARS:
            csv_path = os.path.join(city_dir, f"{city}_{year}.csv")
            parquet_path = os.path.join(city_dir, f"{city}_{year}.parquet")

            if os.path.exists(csv_path) and os.path.exists(parquet_path):
                print(f"Skipping {city} {year}; outputs already exist.")
                continue

            print(f"Starting download for {city} {year}")
            try:
                download_url = submit_download_request_with_fallback(city, year, wkt_points)
                content = wait_and_download_zip(download_url, city, year)
                frame = content_to_dataframe(content)

                frame.to_csv(csv_path, index=False)
                frame.to_parquet(parquet_path, index=False)
                print(f"Saved {csv_path} and {parquet_path}")
            except Exception as exc:
                print(f"Failed for {city} {year}: {exc}")
                failures.append({"city": city, "year": year, "error": str(exc)})
                continue

            # Small delay between API submissions.
            time.sleep(1)

    if failures:
        failures_path = os.path.join(OUTPUT_DIR, "failed_requests.csv")
        pd.DataFrame(failures).to_csv(failures_path, index=False)
        print(f"Completed with {len(failures)} failures. Details: {failures_path}")
    else:
        print(f"Completed downloads for {len(CITIES)} cities x {len(YEARS)} years.")


if __name__ == "__main__":
    main()
