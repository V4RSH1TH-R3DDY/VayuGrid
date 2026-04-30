from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from simulator.config import LoadProfileConfig
from simulator.models import HomeAsset


@dataclass
class ProfileMatrices:
    load_kw: np.ndarray
    pv_kw: np.ndarray
    ev_kw: np.ndarray


@dataclass(frozen=True)
class CityProfile:
    name: str
    base_daily_kwh: float
    summer_day_temp_c: float
    ac_gain_multiplier: float
    evening_peak_multiplier: float
    solar_capacity_factor: float
    ev_arrival_hour: float
    ev_overnight_bias: float


CITY_PROFILES: dict[str, CityProfile] = {
    "bangalore": CityProfile("bangalore", 6.0, 32.0, 0.85, 1.00, 0.84, 19.3, 0.72),
    "chennai": CityProfile("chennai", 7.4, 37.0, 1.28, 0.95, 0.88, 19.8, 0.76),
    "hyderabad": CityProfile("hyderabad", 7.0, 36.0, 1.18, 1.02, 0.87, 19.6, 0.74),
    "delhi": CityProfile("delhi", 7.8, 39.0, 1.35, 1.08, 0.83, 20.1, 0.80),
    "kochi": CityProfile("kochi", 5.8, 33.0, 0.92, 1.04, 0.79, 19.1, 0.68),
    "mumbai": CityProfile("mumbai", 6.7, 34.0, 1.00, 1.03, 0.80, 20.0, 0.77),
}


class LoadProfileLibrary:
    def __init__(
        self,
        config: LoadProfileConfig,
        random_seed: int,
    ) -> None:
        self.config = config
        self.rng = np.random.default_rng(random_seed)
        self.city_profile = CITY_PROFILES.get(
            config.city.lower(),
            CityProfile(
                name=config.city.lower(),
                base_daily_kwh=config.target_daily_kwh,
                summer_day_temp_c=35.0,
                ac_gain_multiplier=1.0,
                evening_peak_multiplier=1.0,
                solar_capacity_factor=0.82,
                ev_arrival_hour=19.5,
                ev_overnight_bias=0.75,
            ),
        )

    def generate_profiles(
        self,
        time_index: pd.DatetimeIndex,
        home_assets: list[HomeAsset],
    ) -> ProfileMatrices:
        if self.config.use_pecan_profiles and self.config.pecan_profile_file:
            load_kw, pv_kw, ev_kw = self._generate_from_pecan(time_index, home_assets)
        else:
            load_kw, pv_kw, ev_kw = self._generate_synthetic_india(time_index, home_assets)

        if self.config.replace_solar_with_nsrdb:
            nsrdb_norm = self._load_nsrdb_normalized_ghi(time_index)
            for idx, asset in enumerate(home_assets):
                pv_kw[:, idx] = (
                    nsrdb_norm
                    * asset.pv_capacity_kw
                    * self.city_profile.solar_capacity_factor
                    if asset.has_solar
                    else 0.0
                )

        return ProfileMatrices(load_kw=load_kw, pv_kw=pv_kw, ev_kw=ev_kw)

    def _generate_synthetic_india(
        self,
        time_index: pd.DatetimeIndex,
        home_assets: list[HomeAsset],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_steps = len(time_index)
        num_homes = len(home_assets)

        load_kw = np.zeros((num_steps, num_homes), dtype=float)
        pv_kw = np.zeros((num_steps, num_homes), dtype=float)
        ev_kw = np.zeros((num_steps, num_homes), dtype=float)

        hours = time_index.hour.to_numpy() + time_index.minute.to_numpy() / 60.0
        is_weekend = time_index.dayofweek.to_numpy() >= 5
        ambient_temp = self._ambient_temperature_curve(time_index)
        festival_flags = self._date_flags(time_index, set(self.config.festival_dates))
        cricket_flags = self._cricket_flags(time_index)

        morning = 0.18 * np.exp(-((hours - 7.5) ** 2) / (2 * 1.4**2))
        afternoon_ac = (
            self.config.afternoon_ac_gain
            * self.city_profile.ac_gain_multiplier
            * np.clip((ambient_temp - self.config.afternoon_ac_threshold_c) / 6.0, 0.0, 1.4)
            * np.exp(-((hours - 15.0) ** 2) / (2 * 2.3**2))
        )
        evening_peak = (
            self.config.evening_peak_gain
            * self.city_profile.evening_peak_multiplier
            * np.exp(-((hours - 20.0) ** 2) / (2 * 1.8**2))
        )
        cricket_spike = (
            self.config.cricket_spike_gain
            * cricket_flags.astype(float)
            * np.exp(-((hours - 21.0) ** 2) / (2 * 1.2**2))
        )
        baseline = 0.22 + morning + afternoon_ac + evening_peak + cricket_spike
        baseline *= np.where(is_weekend, 1.08, 1.0)
        baseline *= np.where(
            festival_flags,
            1.0
            + self.config.festival_spike_gain
            * (0.65 + 0.35 * np.exp(-((hours - 20.0) ** 2) / (2 * 1.7**2))),
            1.0,
        )

        for col, asset in enumerate(home_assets):
            target_daily = self._sample_target_daily_kwh()
            noise = self.rng.normal(loc=0.0, scale=self.config.gaussian_noise_sigma, size=num_steps)
            occupancy_noise = float(self.rng.normal(loc=1.0, scale=0.06))
            raw = np.clip(baseline * occupancy_noise * (1.0 + noise), 0.03, None)

            raw_energy_kwh = raw.sum() / 60.0
            scale = (target_daily * (num_steps / 1440.0)) / max(raw_energy_kwh, 1e-6)
            load_kw[:, col] = raw * scale

            if asset.has_solar:
                solar_shape = np.sin(np.pi * np.clip((hours - 6.0) / 12.0, 0.0, 1.0)) ** 1.8
                cloud_noise = self.rng.normal(loc=1.0, scale=0.10, size=num_steps)
                pv_kw[:, col] = np.clip(
                    asset.pv_capacity_kw
                    * self.city_profile.solar_capacity_factor
                    * solar_shape
                    * cloud_noise,
                    0.0,
                    None,
                )

            if asset.has_ev:
                ev_kw[:, col] = self._ev_profile_india(
                    time_index,
                    asset.ev_daily_kwh,
                    asset.ev_max_charge_kw,
                )

        return load_kw, pv_kw, ev_kw

    def _sample_target_daily_kwh(self) -> float:
        centered = self.rng.normal(loc=self.city_profile.base_daily_kwh, scale=0.7)
        return float(
            np.clip(
                centered,
                self.config.target_daily_kwh_min,
                self.config.target_daily_kwh_max,
            )
        )

    def _ambient_temperature_curve(self, time_index: pd.DatetimeIndex) -> np.ndarray:
        hours = time_index.hour.to_numpy() + time_index.minute.to_numpy() / 60.0
        months = time_index.month.to_numpy()
        seasonal_term = 2.2 * np.sin(((months - 4.0) / 12.0) * 2.0 * np.pi)
        daily_term = 4.6 * np.sin(((hours - 9.0) / 24.0) * 2.0 * np.pi)
        return self.city_profile.summer_day_temp_c + seasonal_term + daily_term

    def _date_flags(self, time_index: pd.DatetimeIndex, marked_dates: set[str]) -> np.ndarray:
        return np.array([ts.date().isoformat() in marked_dates for ts in time_index], dtype=bool)

    def _cricket_flags(self, time_index: pd.DatetimeIndex) -> np.ndarray:
        marked_dates = set(self.config.cricket_match_dates)
        return np.array(
            [
                (ts.date().isoformat() in marked_dates) or (ts.dayofweek >= 5 and 19 <= ts.hour < 23)
                for ts in time_index
            ],
            dtype=bool,
        )

    def _ev_profile_india(
        self,
        time_index: pd.DatetimeIndex,
        daily_kwh: float,
        max_charge_kw: float,
    ) -> np.ndarray:
        output = np.zeros(len(time_index), dtype=float)
        df = pd.DataFrame({"timestamp": time_index})
        df["date"] = df["timestamp"].dt.date

        for day, group in df.groupby("date"):
            day_indices = group.index.to_numpy()
            if len(day_indices) == 0:
                continue

            demand_kwh = max(0.0, daily_kwh * float(self.rng.uniform(0.7, 1.3)))
            if demand_kwh < 0.2:
                continue

            arrival_hour = float(
                np.clip(
                    self.rng.normal(self.city_profile.ev_arrival_hour, 1.1),
                    17.0,
                    23.0,
                )
            )
            start_minutes = int(round(arrival_hour * 60.0))
            minutes_needed = int(np.ceil((demand_kwh / max(max_charge_kw, 1e-6)) * 60.0))
            overnight_target = int(round(minutes_needed * self.city_profile.ev_overnight_bias))

            allocated = 0
            for idx in day_indices:
                ts = time_index[idx]
                minute_of_window = ts.hour * 60 + ts.minute
                if minute_of_window < start_minutes:
                    continue
                if allocated >= minutes_needed:
                    break

                output[idx] = max_charge_kw
                allocated += 1
                if allocated >= overnight_target and ts.hour >= 22:
                    break

            next_day = date.fromisoformat(day.isoformat()) + pd.Timedelta(days=1)
            next_day_mask = (df["timestamp"] >= pd.Timestamp(next_day)) & (
                df["timestamp"] < pd.Timestamp(next_day) + pd.Timedelta(hours=6)
            )
            next_indices = df.index[next_day_mask].to_numpy()
            for idx in next_indices:
                if allocated >= minutes_needed:
                    break
                output[idx] = max_charge_kw
                allocated += 1

        return output

    def _generate_from_pecan(
        self,
        time_index: pd.DatetimeIndex,
        home_assets: list[HomeAsset],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pecan_path = Path(self.config.pecan_profile_file or "")
        if not pecan_path.exists():
            raise FileNotFoundError(f"Pecan profile file not found: {pecan_path}")

        frame = pd.read_csv(
            pecan_path,
            usecols=["timestamp_ist", "home_id", "load_kw", "pv_kw", "ev_kw"],
            low_memory=False,
        )
        frame["timestamp_ist"] = pd.to_datetime(frame["timestamp_ist"], errors="coerce")
        frame = frame.dropna(subset=["timestamp_ist", "home_id"]).sort_values("timestamp_ist")

        available_homes = sorted(frame["home_id"].astype(int).unique().tolist())
        if len(available_homes) < len(home_assets):
            raise RuntimeError("Pecan profile file has fewer homes than requested simulation size")

        selected_homes = self.rng.choice(available_homes, size=len(home_assets), replace=False)
        load_kw = np.zeros((len(time_index), len(home_assets)), dtype=float)
        pv_kw = np.zeros((len(time_index), len(home_assets)), dtype=float)
        ev_kw = np.zeros((len(time_index), len(home_assets)), dtype=float)

        for col, (asset, source_home) in enumerate(zip(home_assets, selected_homes, strict=True)):
            subset = frame[frame["home_id"].astype(int) == int(source_home)][
                ["timestamp_ist", "load_kw", "pv_kw", "ev_kw"]
            ].copy()
            subset = subset.set_index("timestamp_ist").reindex(time_index).interpolate(method="time")
            subset = subset.ffill().bfill().fillna(0.0)

            raw_load = subset["load_kw"].to_numpy(dtype=float)
            raw_pv = subset["pv_kw"].to_numpy(dtype=float)
            raw_ev = subset["ev_kw"].to_numpy(dtype=float)

            target_daily = self._sample_target_daily_kwh()
            raw_energy_kwh = raw_load.sum() / 60.0
            scaled = raw_load * (
                (target_daily * (len(time_index) / 1440.0)) / max(raw_energy_kwh, 1e-6)
            )

            load_kw[:, col] = self._apply_india_shaping(time_index, np.clip(scaled, 0.0, None))
            pv_kw[:, col] = np.clip(raw_pv, 0.0, None) if asset.has_solar else 0.0
            ev_kw[:, col] = np.clip(raw_ev, 0.0, asset.ev_max_charge_kw) if asset.has_ev else 0.0

        return load_kw, pv_kw, ev_kw

    def _apply_india_shaping(self, time_index: pd.DatetimeIndex, load_kw: np.ndarray) -> np.ndarray:
        hours = time_index.hour.to_numpy() + time_index.minute.to_numpy() / 60.0
        ambient_temp = self._ambient_temperature_curve(time_index)
        festival_flags = self._date_flags(time_index, set(self.config.festival_dates))
        cricket_flags = self._cricket_flags(time_index)

        ac_boost = 1.0 + (
            self.config.afternoon_ac_gain
            * self.city_profile.ac_gain_multiplier
            * np.clip((ambient_temp - self.config.afternoon_ac_threshold_c) / 8.0, 0.0, 0.40)
            * np.exp(-((hours - 15.0) ** 2) / (2 * 2.5**2))
        )
        evening_boost = 1.0 + (
            0.45
            * self.city_profile.evening_peak_multiplier
            * np.exp(-((hours - 20.0) ** 2) / (2 * 1.8**2))
        )
        festival_boost = 1.0 + np.where(
            festival_flags,
            self.config.festival_spike_gain
            * np.exp(-((hours - 20.0) ** 2) / (2 * 1.9**2)),
            0.0,
        )
        cricket_boost = 1.0 + np.where(
            cricket_flags,
            self.config.cricket_spike_gain
            * np.exp(-((hours - 21.0) ** 2) / (2 * 1.3**2)),
            0.0,
        )

        shaped = load_kw * ac_boost * evening_boost * festival_boost * cricket_boost
        energy_before = load_kw.sum() / 60.0
        energy_after = shaped.sum() / 60.0
        return shaped * (energy_before / max(energy_after, 1e-9))

    def _load_nsrdb_normalized_ghi(self, time_index: pd.DatetimeIndex) -> np.ndarray:
        city = self.config.city
        year = self.config.year
        nsrdb_path = Path(self.config.nsrdb_data_root) / city / f"{city}_{year}.csv"
        if not nsrdb_path.exists():
            return np.zeros(len(time_index), dtype=float)

        frame = pd.read_csv(nsrdb_path, skiprows=2, low_memory=False)
        required = {"Year", "Month", "Day", "Hour", "Minute", "GHI"}
        if not required.issubset(frame.columns):
            return np.zeros(len(time_index), dtype=float)

        ts = pd.to_datetime(
            {
                "year": pd.to_numeric(frame["Year"], errors="coerce"),
                "month": pd.to_numeric(frame["Month"], errors="coerce"),
                "day": pd.to_numeric(frame["Day"], errors="coerce"),
                "hour": pd.to_numeric(frame["Hour"], errors="coerce"),
                "minute": pd.to_numeric(frame["Minute"], errors="coerce"),
            },
            errors="coerce",
        )
        ghi = pd.to_numeric(frame["GHI"], errors="coerce")

        nsrdb = pd.DataFrame({"timestamp": ts, "ghi": ghi}).dropna(subset=["timestamp", "ghi"])
        nsrdb = nsrdb.set_index("timestamp").sort_index()
        nsrdb = nsrdb[~nsrdb.index.duplicated(keep="first")]
        nsrdb = nsrdb.reindex(time_index).interpolate(method="time").fillna(0.0)

        return (nsrdb["ghi"].to_numpy(dtype=float) / 1000.0).clip(min=0.0)
