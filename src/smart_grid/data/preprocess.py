"""Training-ready profiles with chronological splits and train-only scaling."""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import requests

from smart_grid.config import ROOT, save_json, sha256
from smart_grid.data.download import OPSD_URL, download_opsd
from smart_grid.data.synthetic import generate_profiles

FEATURES = ["solar_cf", "wind_cf", "load_factor"]
OPSD_COLUMNS = {"DE_solar_profile": "solar_cf", "DE_wind_profile": "wind_cf",
                "DE_load_actual_entsoe_transparency": "load_mw"}


def clean_frame(frame: pd.DataFrame, interval_seconds: int = 300,
                max_gap_seconds: int = 3600) -> pd.DataFrame:
    """Mean duplicates, causal forward fill for short gaps, reject long gaps.

    Upsampling is a held-profile assumption, not newly observed high-rate data.
    No backward filling from a future time or test partition is performed.
    """
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = frame.loc[~frame.index.isna()].sort_index()
    frame = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame = frame.mask(frame < 0).groupby(level=0).mean()
    if len(frame) < 3:
        raise ValueError("Not enough valid profile rows")
    native_seconds = float(frame.index.to_series().diff().dropna().median().total_seconds())
    interval = pd.Timedelta(seconds=interval_seconds)
    if native_seconds > interval_seconds:
        # Reindex preserves the original timestamps even when grids are misaligned.
        grid = pd.date_range(frame.index[0], frame.index[-1], freq=interval)
        frame = frame.reindex(frame.index.union(grid)).sort_index()
        last_valid = pd.DataFrame({c: pd.Series(frame.index, index=frame.index).where(frame[c].notna())
                                   for c in frame.columns}).ffill()
        filled = frame.ffill()
        for col in frame:
            age = (pd.Series(frame.index, index=frame.index) - last_valid[col]).dt.total_seconds()
            filled.loc[age > max_gap_seconds, col] = np.nan
        frame = filled.reindex(grid)
    else:
        frame = frame.resample(interval).mean()
        frame = frame.ffill(limit=max(1, int(max_gap_seconds // interval_seconds)))
    # Only an incomplete leading prefix may be removed. Interior gaps remain errors.
    valid_rows = frame.notna().all(axis=1)
    if not valid_rows.any():
        raise ValueError("No timestamps have all required signals")
    frame = frame.loc[valid_rows.idxmax():]
    if frame.isna().any().any():
        raise ValueError("Long missing-data gap: select a different public-data date range")
    frame.index.name = "timestamp"
    return frame


def load_opsd(path: str | Path, start: str, days: int,
              interval_seconds: int = 300) -> pd.DataFrame:
    start_time = pd.Timestamp(start, tz="UTC")
    end_time = start_time + pd.Timedelta(days=days)
    usecols = ["utc_timestamp", *OPSD_COLUMNS]
    pieces = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=20000):
        timestamps = pd.DatetimeIndex(pd.to_datetime(chunk.pop("utc_timestamp"), utc=True, errors="coerce"))
        chunk.index = timestamps
        selected = chunk.loc[(timestamps >= start_time) & (timestamps < end_time)]
        if len(selected):
            pieces.append(selected)
    if not pieces:
        raise ValueError("No OPSD data in requested range (release covers 2015 to mid-2020)")
    return clean_frame(pd.concat(pieces).rename(columns=OPSD_COLUMNS), interval_seconds)


def load_custom(path: str | Path, interval_seconds: int = 300) -> pd.DataFrame:
    """CSV schema: timestamp, solar_mw, wind_mw, load_mw (all MW, UTC)."""
    frame = pd.read_csv(path)
    required = {"timestamp", "solar_mw", "wind_mw", "load_mw"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Custom CSV must contain {sorted(required)}")
    frame = frame.set_index("timestamp")[["solar_mw", "wind_mw", "load_mw"]]
    return clean_frame(frame, interval_seconds)


def split_and_scale(frame: pd.DataFrame, train_fraction: float = 0.7,
                    validation_fraction: float = 0.15) -> tuple[dict[str, pd.DataFrame], dict]:
    n_train = int(len(frame) * train_fraction)
    n_val_end = int(len(frame) * (train_fraction + validation_fraction))
    if min(n_train, n_val_end - n_train, len(frame) - n_val_end) < 10:
        raise ValueError("Need at least ten rows in every chronological partition")
    training = frame.iloc[:n_train]
    scales = {}
    normalized = frame.copy()
    for raw, target, quantile in (("solar_mw", "solar_cf", 0.99),
                                   ("wind_mw", "wind_cf", 0.99),
                                   ("load_mw", "load_factor", 0.5)):
        if raw in normalized:
            scale = float(training[raw].quantile(quantile))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"Positive training scaling factor required for {raw}")
            scales[raw] = scale
            normalized[target] = normalized[raw] / scale
    normalized = normalized[FEATURES]
    clipped = ((normalized[["solar_cf", "wind_cf"]] < 0) |
               (normalized[["solar_cf", "wind_cf"]] > 1)).sum().to_dict()
    normalized[["solar_cf", "wind_cf"]] = normalized[["solar_cf", "wind_cf"]].clip(0, 1)
    if (normalized["load_factor"] <= 0).any() or not np.isfinite(normalized.to_numpy()).all():
        raise ValueError("Nonfinite or nonpositive load data after cleaning")
    splits = {"train": normalized.iloc[:n_train].copy(),
              "validation": normalized.iloc[n_train:n_val_end].copy(),
              "test": normalized.iloc[n_val_end:].copy()}
    return splits, {"train_only_scales": scales, "renewable_rows_clipped": clipped,
                    "custom_scaling_note": "99th percentile MW is a profile normalization proxy, not measured installed capacity"}


def prepare_data(config: dict, source: str = "synthetic", custom_csv: str | Path | None = None,
                 allow_fallback: bool = False, output_dir: str | Path | None = None) -> dict:
    output_dir = Path(output_dir or ROOT / "data/processed")
    data = config["data"]
    actual_source = source
    fallback_reason = None
    raw_path = None
    if source == "synthetic":
        frame = generate_profiles(data["days"], data["interval_seconds"], config["seed"], data["start"])
    elif source == "opsd":
        try:
            raw_path = download_opsd(ROOT / "data/raw/opsd_15min.csv")
            frame = load_opsd(raw_path, data["start"], data["days"], data["interval_seconds"])
        except (requests.RequestException, ValueError, OSError) as error:
            if not allow_fallback:
                raise
            fallback_reason = str(error)
            warnings.warn(f"Public-data loading failed; explicitly enabled synthetic fallback: {error}")
            actual_source = "synthetic_fallback"
            frame = generate_profiles(data["days"], data["interval_seconds"], config["seed"], data["start"])
    elif source == "custom" and custom_csv is not None:
        raw_path = Path(custom_csv)
        frame = load_custom(raw_path, data["interval_seconds"])
    else:
        raise ValueError("Choose synthetic, opsd, or custom with --csv")
    splits, scaling = split_and_scale(frame, data["train_fraction"], data["validation_fraction"])
    output_dir.mkdir(parents=True, exist_ok=True)
    full = pd.concat(list(splits.values()))
    full.to_csv(output_dir / "profiles.csv")
    provenance = {"requested_source": source, "actual_source": actual_source,
                  "source_url": OPSD_URL if actual_source == "opsd" else None,
                  "fallback_reason": fallback_reason, "seed": config["seed"],
                  "profile_interval_seconds": data["interval_seconds"],
                  "fast_variation": "Generated separately in the environment; never measured subsecond data",
                  "scaling": scaling, "partitions": {}}
    for split, rows in splits.items():
        path = output_dir / f"{split}.csv"
        rows.to_csv(path)
        provenance["partitions"][split] = {"rows": len(rows), "start": str(rows.index[0]),
                                            "end": str(rows.index[-1]), "sha256": sha256(path)}
    if raw_path is not None and raw_path.exists():
        provenance["raw_sha256"] = sha256(raw_path)
    save_json(output_dir / "provenance.json", provenance)
    return provenance


def read_profiles(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: smart-grid prepare --source synthetic")
    frame = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    frame.index = pd.to_datetime(frame.index, utc=True)
    if not set(FEATURES).issubset(frame.columns) or len(frame) < 3:
        raise ValueError("Invalid prepared profiles")
    values = frame[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Profiles must be finite and nonnegative")
    if (values[:, :2] > 1).any() or (values[:, 2] <= 0).any():
        raise ValueError("Renewable fractions must be at most one and load must be positive")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("Profile timestamps must increase strictly")
    differences = np.diff(frame.index.asi8)
    if not np.all(differences == differences[0]):
        raise ValueError("Prepared profiles must have a regular time interval")
    return frame[FEATURES]
