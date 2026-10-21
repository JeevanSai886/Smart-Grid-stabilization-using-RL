import numpy as np
import pandas as pd
import pytest

from smart_grid.data.synthetic import generate_profiles
from smart_grid.data.preprocess import clean_frame, split_and_scale, prepare_data, read_profiles, load_opsd
from smart_grid.config import load_config


def test_synthetic_is_seeded_nonnegative_and_dark_at_night():
    a, b = generate_profiles(seed=42), generate_profiles(seed=42)
    pd.testing.assert_frame_equal(a, b)
    assert (a.to_numpy() >= 0).all()
    assert a.loc[a.index.hour < 6, "solar_cf"].eq(0).all()
    assert a.solar_cf.le(1).all() and a.wind_cf.le(1).all()


def test_custom_scaling_cannot_use_test_values():
    index = pd.date_range("2018-01-01", periods=100, freq="h", tz="UTC")
    a = pd.DataFrame({"solar_mw": 1.0, "wind_mw": 1.0, "load_mw": 2.0}, index=index)
    b = a.copy()
    b.iloc[85:] *= 100
    first, _ = split_and_scale(a)
    second, _ = split_and_scale(b)
    pd.testing.assert_frame_equal(first["train"], second["train"])
    assert first["train"].index[-1] < first["validation"].index[0]
    assert first["validation"].index[-1] < first["test"].index[0]


def test_reject_long_gap():
    index = pd.date_range("2018-01-01", periods=100, freq="5min", tz="UTC")
    frame = pd.DataFrame({"load_mw": 1.0}, index=index)
    frame.iloc[20:60] = np.nan
    with pytest.raises(ValueError, match="Long missing"):
        clean_frame(frame)


def test_offline_pipeline(tmp_path):
    meta = prepare_data(load_config(), output_dir=tmp_path)
    assert meta["actual_source"] == "synthetic"
    for split in ("train", "validation", "test"):
        frame = read_profiles(tmp_path / f"{split}.csv")
        assert len(frame) > 10


def test_opsd_csv_timestamp_filter_and_column_mapping(tmp_path):
    timestamps = pd.date_range("2018-01-01", periods=200, freq="15min", tz="UTC")
    path = tmp_path / "opsd.csv"
    pd.DataFrame({"utc_timestamp": timestamps, "DE_solar_profile": 0.2,
                  "DE_wind_profile": 0.4, "DE_load_actual_entsoe_transparency": 40000}).to_csv(path, index=False)
    result = load_opsd(path, "2018-01-02", 1)
    assert set(result.columns) == {"solar_cf", "wind_cf", "load_mw"}
    assert result.index.min() == pd.Timestamp("2018-01-02", tz="UTC")
    assert result.index.max() < pd.Timestamp("2018-01-03", tz="UTC")
    assert result.solar_cf.eq(0.2).all()


def test_network_failure_fallback_is_explicit_and_labeled(tmp_path, monkeypatch):
    import requests
    import smart_grid.data.preprocess as pipeline
    def unavailable(*args, **kwargs):
        raise requests.ConnectionError("Intentional offline test")
    monkeypatch.setattr(pipeline, "download_opsd", unavailable)
    with pytest.raises(requests.ConnectionError):
        prepare_data(load_config(), source="opsd", output_dir=tmp_path)
    with pytest.warns(UserWarning, match="synthetic fallback"):
        metadata = prepare_data(load_config(), source="opsd", allow_fallback=True, output_dir=tmp_path)
    assert metadata["actual_source"] == "synthetic_fallback"
    assert "Intentional offline" in metadata["fallback_reason"]
