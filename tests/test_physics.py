import numpy as np
import pytest

from smart_grid.config import load_config
from smart_grid.grid.battery import Battery
from smart_grid.grid.frequency import FrequencyModel
from smart_grid.grid.topology import Feeder


def test_battery_limits_efficiency_and_cycles():
    c = load_config()["battery"]
    b = Battery(c, c["soc_max"])
    assert b.dispatch(-100, 3600) == 0
    delivered = b.dispatch(100, 3600)
    stored = c["capacity_mwh"] * (c["soc_max"] - c["soc_min"])
    assert delivered == pytest.approx(stored * c["discharge_efficiency"])
    assert b.soc == pytest.approx(c["soc_min"])
    assert b.dispatch(100, 1) == 0
    consumed = -b.dispatch(-100, 3600)
    assert consumed == pytest.approx(stored / c["charge_efficiency"])
    assert b.equivalent_cycles == pytest.approx(c["soc_max"] - c["soc_min"])


def test_frequency_equilibrium_deficit_and_source_limit():
    c = load_config()["frequency"]
    f = FrequencyModel(c)
    f.reset(2.0, 3.0)
    f.advance(2.0, 3.0, 5.0)
    assert abs(f.deviation) < 1e-12
    f.advance(2.5, 2.0, 1.0)
    assert f.deviation < 0
    assert 0 <= f.mechanical_mw <= 2.0


def test_frequency_integration_converges():
    c = load_config()["frequency"]
    coarse, fine = FrequencyModel(c), FrequencyModel({**c, "integration_dt_s": 0.005})
    for f in (coarse, fine):
        f.reset(2, 3)
        f.advance(2.4, 3, 2)
    assert coarse.deviation == pytest.approx(fine.deviation, abs=0.015)


def test_ac_power_balance_voltage_response_and_priorities():
    feeder = Feeder(load_config()["environment"])
    a = feeder.solve(1.1, 0.2, 0.2, 0.0, 0)
    b = feeder.solve(1.1, 0.2, 0.2, 0.7, 0)
    assert a["converged"] and b["converged"]
    assert b["voltages"].min() > a["voltages"].min()
    assert b["source_electrical_mw"] + b["solar_mw"] + b["wind_mw"] + 0.7 == pytest.approx(
        b["served_mw"] + b["losses_mw"], abs=2e-5)
    for level in (1, 2):
        p = feeder.solve(1.1, 0.2, 0.2, 0, level)
        assert p["shed_by_tier_mw"][0] == 0
        assert p["shed_by_tier_mw"][2] > 0
        assert p["served_mw"] < a["served_mw"]

