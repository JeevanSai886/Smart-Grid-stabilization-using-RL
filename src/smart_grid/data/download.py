"""Streaming downloads with bounded size and atomic replacement."""
from __future__ import annotations

from pathlib import Path
import requests

OPSD_URL = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_15min_singleindex.csv"


def download_opsd(destination: str | Path, max_bytes: int = 180_000_000) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1000:
        return path
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        with requests.get(OPSD_URL, stream=True, timeout=(20, 90)) as response:
            response.raise_for_status()
            if "text/html" in response.headers.get("Content-Type", ""):
                raise ValueError("Expected CSV, received HTML")
            received = 0
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        received += len(chunk)
                        if received > max_bytes:
                            raise ValueError("Download exceeds configured size limit")
                        handle.write(chunk)
        with temporary.open(encoding="utf-8") as handle:
            header = handle.readline()
        if "utc_timestamp" not in header:
            raise ValueError("OPSD timestamp column missing from download")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path
