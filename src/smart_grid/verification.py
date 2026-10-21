"""Audits saved artifacts only; does not retrain policies or rerun simulations."""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from smart_grid.config import ROOT, save_json, sha256
from smart_grid.controllers.registry import ORDER, LEARNED, AGENTS
from smart_grid.metrics import summarize_episode
from smart_grid.plotting import matched_cells, overall_summary


def verify_run(run_dir):
    run = Path(run_dir)
    config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    directory = run / "evaluation"
    rows = pd.read_csv(directory / "episodes.csv")
    matched_cells(rows, ["delay_seconds", "fluctuation_scale", "episode_seed"])
    expected = len(ORDER) * config["evaluation"]["episodes"] * len(config["evaluation"]["delay_steps"]) * len(config["evaluation"]["fluctuation_scales"])
    if len(rows) != expected:
        raise AssertionError("Missing or extra test episodes.")
    validation = pd.read_csv(run / "validation_comparison.csv")
    matched_cells(validation, ["training_timesteps", "episode_seed"])
    target = config["training"]["total_timesteps"]
    assert validation.training_timesteps.max() == target
    for controller in LEARNED:
        metadata = json.loads((run / "models" / controller / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["actual_timesteps"] == metadata["requested_timesteps"] == target
        assert metadata["status"] == "complete"
        assert (run / "models" / controller / metadata["selected_model"]).is_file()
    references = {}
    action_rows = []
    for row in rows.itertuples():
        path = directory / "traces" / f"{row.controller}_d{row.delay_steps}_f{row.fluctuation_scale:g}_seed{row.episode_seed}.csv"
        trace = pd.read_csv(path, dtype={"blackout_reason": "string"})
        assert len(trace) == config["evaluation"]["episode_steps"]
        assert np.isfinite(trace.select_dtypes(include=[np.number])).all().all()
        expected_age = np.minimum(trace.step, row.delay_steps) * config["environment"]["control_dt_s"]
        np.testing.assert_allclose(trace.observation_age_s, expected_age)
        np.testing.assert_allclose(trace.reward, trace.filter(regex="^reward_").sum(axis=1), atol=1e-8)
        measured = summarize_episode(trace)
        for key, value in measured.items():
            if not np.isclose(getattr(row, key), value, rtol=1e-8, atol=1e-8):
                raise AssertionError(f"Metric does not match trace: {path.name}, {key}")
        assert trace.soc.between(config["battery"]["soc_min"] - 1e-10, config["battery"]["soc_max"] + 1e-10).all()
        assert trace.unserved_mw.ge(-1e-9).all()
        assert trace.shed_mw.le(trace.unserved_mw + 1e-9).all()
        first, b = trace.iloc[0], config["battery"]
        power = first.battery_power_mw
        change = (-power / b["discharge_efficiency"] if power >= 0 else -power * b["charge_efficiency"]) * first.dt_s / 3600
        initial_soc = first.soc - change / b["capacity_mwh"]
        key = (row.episode_seed, row.fluctuation_scale)
        signals = trace[["demand_mw", "source_capacity_mw"]].to_numpy()
        if key not in references:
            references[key] = (signals, initial_soc, trace.timestamp.tolist())
        else:
            np.testing.assert_allclose(signals, references[key][0], atol=1e-9)
            assert np.isclose(initial_soc, references[key][1], atol=1e-9)
            assert trace.timestamp.tolist() == references[key][2]
        counts = trace.groupby(["battery_action", "shedding_action"]).size()
        action_rows.append({"controller": row.controller, "delay_seconds": row.delay_seconds,
                            "episode_seed": row.episode_seed, "distinct_action_pairs": len(counts),
                            "dominant_action_fraction": float(counts.max()/len(trace))})
    pd.DataFrame(action_rows).to_csv(directory / "action_diagnostics.csv", index=False)
    _, expected_summary = overall_summary(rows)
    saved_summary = pd.read_csv(directory / "controller_summary.csv", index_col=0)
    pd.testing.assert_frame_equal(saved_summary, expected_summary, check_exact=False, rtol=1e-8)
    manifest = json.loads((directory / "figures/manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["controllers_on_each_comparison_axes"]) == set(ORDER)
    for name in manifest["figures"]:
        png = directory / "figures" / name
        assert png.stat().st_size > 0
        svg = ET.parse(png.with_suffix(".svg"))
        texts = " ".join(node.text or "" for node in svg.iter() if node.tag.endswith("}text"))
        for controller in ORDER:
            assert AGENTS[controller]["label"] in texts, f"Missing visible legend label in {name}"
    tags = {}
    for controller in ORDER:
        accumulator = EventAccumulator(str(run / "tensorboard/agents" / controller), size_guidance={"scalars": 0})
        accumulator.Reload()
        tags[controller] = set(accumulator.Tags().get("scalars", []))
        if not tags[controller]:
            raise AssertionError(f"No scalar events for {controller}")
    common = set.intersection(*tags.values())
    shared_prefixes = ("validation/", "test_delay/", "test_overall/", "grid/")
    for controller in ORDER:
        requested = {tag for tag in tags[controller] if tag.startswith(shared_prefixes)}
        assert requested <= common, f"Some common metrics are missing from a controller: {controller}"
    for metric in ("cumulative_reward", "stability_violation_duration_s", "blackout_duration_s",
                   "blackout_events_per_hour", "shed_mwh", "battery_cycles"):
        assert f"validation/{metric}/mean" in common
    original = json.loads((run / "input_manifest.json").read_text(encoding="utf-8"))
    for split, digest in original["data_sha256"].items():
        assert sha256(ROOT / "data/processed" / f"{split}.csv") == digest
    for name, digest in original["preserved_source_sha256"].items():
        assert sha256(ROOT / name) == digest
    result = {"status": "passed", "test_episodes_checked": len(rows),
              "test_steps_checked": len(rows)*config["evaluation"]["episode_steps"],
              "actual_training_steps_per_ppo": target, "tensorboard_runs": list(ORDER),
              "common_scalar_tags": sorted(common), "figures": manifest["figures"],
              "validation_reference_note": "Rule-Based and Idle reuse measured fixed-policy validation references.",
              "limitations": "Artifact checks do not prove policy convergence or scientific superiority."}
    destination = run / "verification.json"
    save_json(destination, result)
    return destination

