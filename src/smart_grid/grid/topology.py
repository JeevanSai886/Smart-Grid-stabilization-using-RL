from __future__ import annotations

import numpy as np
import pandapower as pp
import pandapower.networks as pn
from pandapower.auxiliary import LoadflowNotConverged


class Feeder:
    """Modified Baran-Wu feeder, balanced AC power flow and explicit tier shares."""
    def __init__(self, config: dict):
        self.config = config
        self.net = pn.case33bw()
        self.net.ext_grid.loc[:, "vm_pu"] = config["slack_voltage_pu"]
        self.net.line.loc[:, "max_i_ka"] = config["line_rating_ka"]
        self.base_p = self.net.load.p_mw.to_numpy(copy=True) * config["load_scale"]
        self.base_q = self.net.load.q_mvar.to_numpy(copy=True) * config["load_scale"]
        self.base_total_mw = float(self.base_p.sum())
        buses = self.net.load.bus.to_numpy()
        critical = 0.25 + 0.05 * (buses % 3)
        medium = 0.30 + 0.025 * (buses % 2)
        self.tier_shares = np.column_stack([critical, medium, 1 - critical - medium])
        self.solar = [pp.create_sgen(self.net, b, p_mw=0.0, name=f"PV bus {b}") for b in config["solar_buses"]]
        self.wind = [pp.create_sgen(self.net, b, p_mw=0.0, name=f"Wind bus {b}") for b in config["wind_buses"]]
        self.battery_index = pp.create_sgen(self.net, config["battery_bus"], p_mw=0.0, name="Battery: discharge positive")
        self.last_ok = False

    def solve(self, load_factor: float, solar_mw: float, wind_mw: float,
              battery_mw: float, shedding_level: int) -> dict:
        retained = np.ones(3)
        if shedding_level >= 1:
            retained[2] = 0
        if shedding_level >= 2:
            retained[1] = 0
        factors = self.tier_shares @ retained
        requested_p = self.base_p * load_factor
        tier_shed = (requested_p[:, None] * self.tier_shares * (1 - retained)).sum(axis=0)
        self.net.load.loc[:, "p_mw"] = requested_p * factors
        self.net.load.loc[:, "q_mvar"] = self.base_q * load_factor * factors
        self.net.sgen.loc[self.solar, "p_mw"] = solar_mw / len(self.solar)
        self.net.sgen.loc[self.wind, "p_mw"] = wind_mw / len(self.wind)
        self.net.sgen.loc[self.battery_index, "p_mw"] = battery_mw
        converged = self._run()
        curtailed = 0.0
        if converged and self.net.res_ext_grid.p_mw.sum() < 0:
            # Islanded source cannot absorb sustained reverse power. Autonomous
            # renewable curtailment is identical for every controller.
            if self.config["mode"] == "islanded":
                available = solar_mw + wind_mw
                for _ in range(3):
                    reverse = max(0.0, -float(self.net.res_ext_grid.p_mw.sum()))
                    if reverse < 1e-6 or available <= 0:
                        break
                    reduction = min(available, reverse * 1.01 + 1e-6)
                    ratio = (available - reduction) / available
                    self.net.sgen.loc[self.solar + self.wind, "p_mw"] *= ratio
                    available -= reduction
                    curtailed += reduction
                    converged = self._run()
                    if not converged:
                        break
        demand = float(requested_p.sum())
        if converged:
            volts = self.net.res_bus.vm_pu.to_numpy(copy=True)
            loading = np.nan_to_num(self.net.res_line.loading_percent.to_numpy(), nan=0.0) / 100
            losses = float(self.net.res_line.pl_mw.sum())
            electrical = float(self.net.res_ext_grid.p_mw.sum())
        else:
            volts = np.zeros(len(self.net.bus))
            loading = np.zeros(len(self.net.line))
            losses = 0.0
            electrical = 0.0
        return {"converged": converged, "voltages": volts, "line_loading": loading,
                "losses_mw": losses, "source_electrical_mw": electrical,
                "demand_mw": demand, "served_mw": demand - float(tier_shed.sum()),
                "shed_by_tier_mw": tier_shed, "curtailed_mw": curtailed,
                "solar_mw": float(self.net.sgen.loc[self.solar, "p_mw"].sum()),
                "wind_mw": float(self.net.sgen.loc[self.wind, "p_mw"].sum())}

    def _run(self) -> bool:
        # Reuse voltage initialization only; changed injections are recomputed.
        for algorithm, init in (("bfsw", "results" if self.last_ok else "flat"), ("nr", "flat")):
            try:
                pp.runpp(self.net, algorithm=algorithm, init=init, numba=True,
                         max_iteration=50, tolerance_mva=1e-7, calculate_voltage_angles=False)
                if self.net.converged and np.isfinite(self.net.res_bus.vm_pu).all():
                    self.last_ok = True
                    return True
            except LoadflowNotConverged:
                pass
        self.last_ok = False
        return False

