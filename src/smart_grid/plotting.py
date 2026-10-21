"""Five-controller comparisons: every comparison axes uses the same five labels."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

from smart_grid.config import save_json, sha256
from smart_grid.controllers.registry import AGENTS, ORDER, require_five

# Values, winners and error bars are computed from saved measurements.
METRICS = {
    "cumulative_reward": ("Episode reward", "Episode reward", True, 1.0, "reward_comparison"),
    "stability_violation_duration_s": ("Stability violation duration", "Violation duration per episode (s)", False, 1.0, "stability_violations"),
    "blackout_duration_s": ("Modeled blackout duration", "Blackout duration per episode (s)", False, 1.0, "blackout_duration"),
    "blackout_events_per_hour": ("Modeled blackout frequency", "Blackout events per simulated hour", False, 1.0, "blackout_frequency"),
    "shed_mwh": ("Voluntary load shedding", "Load shed per episode (kWh)", False, 1000.0, "load_shed"),
    "unserved_mwh": ("Total unserved energy", "Unserved energy per episode (kWh)", False, 1000.0, "unserved_energy"),
    "battery_cycles": ("Battery usage", "Equivalent full cycles per episode", False, 1.0, "battery_cycles"),
    "stable_fraction": ("Stable fraction", "Stable time per episode (%)", True, 100.0, "stable_fraction"),
}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10, "legend.fontsize": 8.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "axes.edgecolor": "#7B8791",
        "grid.color": "#DCE1E6", "grid.linewidth": 0.6,
        "text.color": "#25313B", "axes.labelcolor": "#25313B",
        "svg.fonttype": "none", "savefig.facecolor": "white",
    })


def handles(bars=False):
    result = []
    for key in ORDER:
        s = AGENTS[key]
        if bars:
            result.append(Patch(facecolor=s["color"], edgecolor="white",
                                hatch="//" if key in {"idle_agent", "rule_based"} else None, label=s["label"]))
        else:
            result.append(Line2D([], [], color=s["color"], ls=s["linestyle"], lw=1.8,
                                 marker=s["marker"], markerfacecolor="none", label=s["label"]))
    return result


def figure(title, subtitle, bars=False, height=5.2):
    fig, ax = plt.subplots(figsize=(7.2, height))
    fig.subplots_adjust(left=0.23 if bars else 0.15, right=0.83 if bars else 0.97,
                        top=0.74, bottom=0.24)
    fig.text(0.045, 0.96, title, fontsize=13, weight="bold")
    fig.text(0.045, 0.915, subtitle, fontsize=9)
    fig.legend(handles=handles(bars), loc="upper center", bbox_to_anchor=(0.53, 0.895),
               ncol=3, frameon=False)
    ax.grid(axis="x" if bars else "y")
    ax.set_axisbelow(True)
    return fig, ax


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white")
    fig.savefig(path.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def line(ax, x, y, key, error=None, sparse=False):
    s = AGENTS[key]
    options = {}
    if sparse:
        options["markevery"] = (ORDER.index(key) * 7, max(1, len(x) // 8))
    ax.plot(x, y, color=s["color"], ls=s["linestyle"], marker=s["marker"],
            markerfacecolor="none", markeredgewidth=1.2, markersize=6,
            lw=1.6, label=s["label"], **options)
    if error is not None and pd.Series(error).notna().any():
        ax.errorbar(x, y, yerr=error, fmt="none", ecolor=s["color"], capsize=2.5, elinewidth=0.8)


def matched_cells(frame, keys):
    require_five(frame)
    if frame.duplicated(["controller", *keys]).any():
        raise ValueError("Duplicate controller/scenario rows; refusing an unequal comparison.")
    for _, group in frame.groupby(keys, dropna=False):
        if set(group.controller) != set(ORDER):
            raise ValueError("A scenario is missing a controller; no rows will be silently dropped.")


def overall_summary(frame):
    per_seed = frame.groupby(["controller", "episode_seed"])[list(METRICS)].mean().reset_index()
    summary = per_seed.groupby("controller")[list(METRICS)].agg(["mean", "std", "count"]).reindex(ORDER)
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return per_seed, summary


def metric_plots(frame, destination):
    per_seed, summary = overall_summary(frame)
    per_seed.to_csv(destination.parent / "controller_seed_means.csv", index=False)
    summary.to_csv(destination.parent / "controller_summary.csv")
    n = frame.episode_seed.nunique()
    for metric, (title, unit, higher, scale, filename) in METRICS.items():
        mean, sd = summary[f"{metric}_mean"] * scale, summary[f"{metric}_std"] * scale
        best = mean.max() if higher else mean.min()
        winners = np.isclose(mean.to_numpy(), best, rtol=1e-9, atol=1e-10)
        direction = "Higher is better" if higher else "Lower is better"
        fig, ax = figure(title, f"{direction} | Mean +/- SD across {n} matched scenario seeds", bars=True)
        for i, key in enumerate(ORDER):
            ax.barh(i, mean[key], color=AGENTS[key]["color"], height=0.58,
                    hatch="//" if key in {"idle_agent", "rule_based"} else None,
                    edgecolor="#26313A" if winners[i] else "white", linewidth=1.5 if winners[i] else 0.7)
            if pd.notna(sd[key]):
                ax.errorbar(mean[key], i, xerr=sd[key], fmt="none", ecolor="#24313A", capsize=3, elinewidth=0.8)
            precision = 3 if metric == "battery_cycles" else 1
            ax.text(1.035, i, f"{mean[key]:.{precision}f}" + (" *" if winners[i] else ""),
                    transform=ax.get_yaxis_transform(), va="center", fontsize=9,
                    weight="bold" if winners[i] else "normal")
        lower = min(0.0, float((mean - sd.fillna(0)).min()))
        upper = max(0.0, float((mean + sd.fillna(0)).max()))
        if lower == upper:
            upper = 1.0
        if metric == "stable_fraction":
            upper = max(100.0, upper)
        ax.set_xlim(lower * 1.10, upper * 1.08)
        ax.set_yticks(range(len(ORDER)), [AGENTS[c]["label"] for c in ORDER])
        ax.invert_yaxis()
        ax.set_xlabel(unit)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        fig.text(0.85, 0.76, "Mean", fontsize=8.5)
        fig.text(0.045, 0.12, "Each seed is averaged across the full delay/intensity sweep before computing SD.", fontsize=8)
        fig.text(0.045, 0.075, "* Best observed mean; all ties retained. Hatching identifies non-learning baselines.", fontsize=8)
        note = "Descriptive SD may extend beyond physical bounds; it is not a confidence interval."
        if metric == "battery_cycles":
            note = "Low battery use alone does not establish good grid stabilization."
        elif metric == "stability_violation_duration_s":
            note = "Outages are reported separately; stable fraction treats outage time as unstable."
        fig.text(0.045, 0.033, note, fontsize=8)
        save(fig, destination / f"{filename}.png")
    return summary


def delay_plot(frame, destination):
    intensities = sorted(frame.fluctuation_scale.unique())
    for intensity in intensities:
        subset = frame[frame.fluctuation_scale == intensity]
        grouped = subset.groupby(["controller", "delay_seconds"]).stability_violation_duration_s.agg(["mean", "std"])
        fig, ax = figure("Delay robustness: all five controllers",
                         f"Lower violation duration is better | Fluctuation scale {intensity:g} | Mean +/- SD")
        for key in ORDER:
            series = grouped.loc[key].sort_index()
            line(ax, series.index, series["mean"], key, series["std"])
        ax.set_xticks(sorted(subset.delay_seconds.unique()))
        ax.set_xlabel("Communication delay (s)")
        ax.set_ylabel("Stability violation duration (s)")
        low = min(0.0, float((grouped["mean"] - grouped["std"].fillna(0)).min()))
        ax.set_ylim(bottom=low * 1.1)
        fig.text(0.045, 0.12, "All five controllers share these axes and the same matched test scenarios.", fontsize=8.5)
        fig.text(0.045, 0.07, "Coincident curves indicate equal measured values; no offsets separate them.", fontsize=8.5)
        fig.text(0.045, 0.03, "SD describes scenario variability; one training seed cannot establish superiority.", fontsize=8)
        suffix = "" if len(intensities) == 1 else f"_f{intensity:g}"
        save(fig, destination / f"delay_robustness{suffix}.png")


def reward_curve(run_dir, destination, budget):
    path = run_dir / "validation_comparison.csv"
    if not path.exists():
        raise FileNotFoundError("Run the full pipeline to create matched validation rewards for all five controllers.")
    frame = pd.read_csv(path)
    matched_cells(frame, ["training_timesteps", "episode_seed"])
    grouped = frame.groupby(["controller", "training_timesteps"]).cumulative_reward.agg(["mean", "std"])
    fig, ax = figure("Reward versus PPO training timesteps",
                     "Same held-out validation scenarios for all five controllers | Higher is better", height=5.4)
    for key in ORDER:
        series = grouped.loc[key].sort_index()
        # Every actual validation episode is retained as a faint, unshifted point.
        raw = frame[frame.controller == key]
        ax.scatter(raw.training_timesteps, raw.cumulative_reward, s=13,
                   color=AGENTS[key]["color"], alpha=0.28, marker=AGENTS[key]["marker"])
        line(ax, series.index, series["mean"], key, series["std"])
    ax.set_xlim(0, budget)
    ax.set_xticks(sorted(frame.training_timesteps.unique()))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_xlabel("PPO training timesteps")
    ax.set_ylabel("Validation episode reward")
    fig.text(0.045, 0.13, "Lines: checkpoint means +/- SD. Faint points: actual episode returns. No smoothing.", fontsize=8)
    fig.text(0.045, 0.085, "Rule-Based and Idle: measured fixed references reused at each checkpoint; neither trains.", fontsize=8)
    fig.text(0.045, 0.04, "Raw PPO training episodes are saved separately in each model's train.monitor.csv.", fontsize=8)
    save(fig, destination / "reward_curve.png")



def trajectory_plot(frame, directory, config):
    delay = min(frame.delay_steps.unique(), key=lambda value: (abs(value * config["environment"]["control_dt_s"] - 3), value))
    intensity = min(frame.fluctuation_scale.unique(), key=lambda value: (abs(value - 1), value))
    seed = int(config["evaluation"]["seeds"][0])
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 9.0), sharex=True)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.80, bottom=0.13, hspace=0.39)
    fig.text(0.045, 0.965, "Held-out trajectory: all five controllers", size=13, weight="bold")
    seconds = delay * config["environment"]["control_dt_s"]
    fig.text(0.045, 0.932, f"First configured seed {seed} | delay {seconds:g} s | fluctuation scale {intensity:g}", size=9)
    fig.legend(handles=handles(), loc="upper center", bbox_to_anchor=(0.53, 0.91), ncol=3, frameon=False)
    selected = []
    for key in ORDER:
        path = directory / "traces" / f"{key}_d{delay}_f{intensity:g}_seed{seed}.csv"
        trace = pd.read_csv(path)
        selected.append({"controller": key, "trace": path.name, "sha256": sha256(path)})
        for ax, metric in zip(axes, ("voltage_min_pu", "frequency_deviation_hz", "soc", "shed_mw")):
            values = trace[metric] * (100 if metric == "soc" else 1)
            if metric == "frequency_deviation_hz":
                # Disconnected frequency is undefined, not a genuine zero deviation.
                values = values.mask(trace.blackout.astype(bool))
                blocks = trace.blackout.ne(trace.blackout.shift(fill_value=False)).cumsum()
                for _, block in trace.groupby(blocks):
                    if bool(block.blackout.iloc[0]):
                        ax.axvspan(block.time_s.iloc[0] - block.dt_s.iloc[0], block.time_s.iloc[-1],
                                   color=AGENTS[key]["color"], alpha=0.06, lw=0)
            line(ax, trace.time_s, values, key, sparse=True)
    env, battery = config["environment"], config["battery"]
    limits = ((env["voltage_min_pu"], env["voltage_max_pu"]),
              (-env["frequency_limit_hz"], env["frequency_limit_hz"]),
              (100 * battery["soc_min"], 100 * battery["soc_max"]))
    for ax, pair in zip(axes, limits):
        for value in pair:
            ax.axhline(value, color="#A33131", ls=(0, (3, 3)), lw=0.85)
    for ax, title, unit in zip(axes, ("Minimum bus voltage", "Frequency deviation", "Battery SOC", "Voluntary load shedding"),
                               ("Voltage (p.u.)", "Deviation (Hz)", "SOC (%)", "Shed load (MW)")):
        ax.set_title(title, loc="left", size=10, weight="bold")
        ax.set_ylabel(unit)
        ax.grid(axis="y")
        ax.set_xlim(0, config["evaluation"]["episode_steps"] * env["control_dt_s"])
    axes[0].set_title(f"Limits: {limits[0][0]:.2f}-{limits[0][1]:.2f} p.u.", loc="right", size=8, color="#A33131")
    axes[1].set_title(f"Limits: +/-{env['frequency_limit_hz']:g} Hz", loc="right", size=8, color="#A33131")
    axes[2].set_title(f"Limits: {limits[2][0]:g}-{limits[2][1]:g}%", loc="right", size=8, color="#A33131")
    axes[2].set_ylim(0, 100)
    axes[3].set_ylim(bottom=0)
    axes[-1].set_xlabel("Elapsed test time (s)")
    fig.text(0.045, 0.073, "Red dashed lines: safety limits. Frequency gaps/tinted bands: disconnected intervals.", size=8)
    fig.text(0.045, 0.039, "No outcome-based selection. All five share each panel; overlapping traces are preserved.", size=8)
    save_json(directory / "trajectory_selection.json", {
        "selection": "First configured test seed, delay nearest 3 s, fluctuation nearest 1; never ranked by outcomes",
        "sources": selected})
    save(fig, directory / "figures/example_trajectory.png")


def paired_age_table(frame):
    values = frame.pivot(index=["delay_seconds", "fluctuation_scale", "episode_seed"],
                         columns="controller", values="stability_violation_duration_s")
    if values.isna().any().any():
        raise ValueError("Missing paired observations.")
    differences = (values["delay_aware_agent"] - values["delay_unaware_agent"]).rename("aware_minus_no_age_s").reset_index()
    per_seed = differences.groupby(["delay_seconds", "episode_seed"]).aware_minus_no_age_s.mean()
    summary = per_seed.groupby(level="delay_seconds").agg(["mean", "std", "count"]).reset_index()
    return differences, summary


def make_plots(evaluation_dir, models_dir=None):
    directory = Path(evaluation_dir)
    frame = pd.read_csv(directory / "episodes.csv")
    matched_cells(frame, ["delay_seconds", "fluctuation_scale", "episode_seed"])
    config = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))["config"]
    destination = directory / "figures"
    style()
    metric_plots(frame, destination)
    delay_plot(frame, destination)
    reward_curve(directory.parent, destination, config["training"]["total_timesteps"])
    trajectory_plot(frame, directory, config)
    pairs, summary = paired_age_table(frame)
    pairs.to_csv(directory / "age_ablation_paired.csv", index=False)
    summary.to_csv(directory / "age_ablation_summary.csv", index=False)
    paths = sorted(destination.glob("*.png"))
    save_json(destination / "manifest.json", {
        "controllers_on_each_comparison_axes": list(ORDER),
        "formats": ["PNG 300 dpi", "SVG"], "figure_width_inches": 7.2,
        "episodes_sha256": sha256(directory / "episodes.csv"),
        "validation_sha256": sha256(directory.parent / "validation_comparison.csv"),
        "figures": [path.name for path in paths],
        "uncertainty": "SD across scenario seeds; comparison bars first average each seed's delay sweep",
        "reward_curve": "Actual matched validation rewards; untrained baseline references explicitly reused",
        "styling": AGENTS})
    return paths


def write_results_summary(evaluation_dir, demo=False):
    directory = Path(evaluation_dir)
    frame = pd.read_csv(directory / "episodes.csv")
    config = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))["config"]
    _, stats = overall_summary(frame)
    columns = ("cumulative_reward", "stability_violation_duration_s", "blackout_duration_s",
               "blackout_events_per_hour", "shed_mwh", "unserved_mwh", "battery_cycles", "stable_fraction")
    labels = ("Reward", "Violation s", "Blackout s", "Blackouts/h", "Shed kWh", "Unserved kWh", "Cycles", "Stable %")
    lines = ["# Delay-aware smart-grid experiment", "",
             "Generated from this run's saved measurements. No fixed performance numbers are embedded in this report.", "",
             "## Experimental setup", "",
             f"- Training budget: {config['training']['total_timesteps']:,} steps for each of three PPO variants.",
             f"- Training seed: {config['seed']}; test seeds: {config['evaluation']['seeds']}.",
             "- Rule-Based and Idle have no learned parameters and are not trained.",
             f"- Test episodes: {config['evaluation']['episode_steps']} steps at {config['environment']['control_dt_s']:g} s per step.",
             f"- Delays (seconds): {sorted(frame.delay_seconds.unique().tolist())}.",
             f"- Fluctuation scales: {config['evaluation']['fluctuation_scales']}.",
             "- Same synthetic data/processed partitions, AC feeder, aggregate frequency model, reward, actions, initial SOC and scenario seeds.",
             "- Model selection uses fixed validation cases only. Test episodes are retained regardless of outcome.", "",
             "## Complete comparison", "",
             "Entries are mean +/- SD across per-seed sweep averages. SD is descriptive, not a confidence interval.", "",
             "| Controller | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
    for controller in ORDER:
        values = []
        for metric in columns:
            scale = 1000 if metric in {"shed_mwh", "unserved_mwh"} else 100 if metric == "stable_fraction" else 1
            mean, sd = stats.loc[controller, f"{metric}_mean"] * scale, stats.loc[controller, f"{metric}_std"] * scale
            digits = 3 if metric == "battery_cycles" else 2
            values.append(f"{mean:.{digits}f} +/- {sd:.{digits}f}" if pd.notna(sd) else f"{mean:.{digits}f}; SD unavailable")
        lines.append("| " + AGENTS[controller]["label"] + " | " + " | ".join(values) + " |")
    lines += ["", "## Observed rankings", ""]
    for metric, (_, _, higher, _, _) in METRICS.items():
        means = stats[f"{metric}_mean"]
        best = means.max() if higher else means.min()
        winners = [AGENTS[key]["label"] for key in ORDER if np.isclose(means[key], best, rtol=1e-9, atol=1e-10)]
        lines.append(f"- {METRICS[metric][0]}: {', '.join(winners)} (best observed mean, including ties).")
    pairs, summary = paired_age_table(frame)
    lines += ["", "## Measurement-age research question", "",
              "Difference = Aware minus No-Age violation duration. Negative favors Aware; positive favors No-Age.", "",
              "| Delay s | Mean difference s | SD s | Scenario seeds |", "|---|---:|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"| {row.delay_seconds:g} | {row.mean:.3f} | {row.std:.3f} | {row.count} |")
    difference = float(pairs.aware_minus_no_age_s.mean())
    if np.allclose(pairs.aware_minus_no_age_s, 0, atol=1e-10, rtol=0):
        conclusion = "The paired durations are identical; this run provides no measured age-awareness advantage."
    elif difference < 0:
        conclusion = f"Aware has {abs(difference):.3f} seconds lower mean violation duration in this suite; statistical superiority is not established."
    elif difference > 0:
        conclusion = f"No-Age has {difference:.3f} seconds lower mean violation duration in this suite; this does not support an overall Aware advantage on that metric."
    else:
        conclusion = "Average duration is tied despite individual scenario differences."
    lines += ["", conclusion, "", "## Interpretation and limitations", "",
              "- Idle is a passive reference, not a guaranteed worst performer on every metric.",
              "- Low battery use, low shedding and high stability can conflict; there need not be one universal winner.",
              "- A short budget, one training seed and a small scenario set cannot prove convergence or algorithm-level generalization.",
              "- A constant-action policy can produce a flat delay curve without learning to adapt to delay.",
              "- Power flow is balanced AC coupled to aggregate swing/governor dynamics, not a full transient grid simulation.",
              "- Synthetic profile fluctuations and communication delays are simulated, not measured field telemetry.",
              "- Outages and stability violations are separate metrics; stable fraction treats outages as unstable.",
              "- The all-five reward curve uses matched validation returns. Baselines reuse their measured fixed reference; PPO training-only returns remain in monitor CSVs.",
              "- TensorBoard shares validation, test and physical-signal tags across all five runs. PPO-only optimizer diagnostics remain in optimizer_logs/progress.csv.", "",
              "## Figures", "", "Each comparison plot includes all five controllers with a common legend. PNGs are 300 dpi; SVGs are scalable.", ""]
    for path in sorted((directory / "figures").glob("*.png")):
        lines += [f"![{path.stem.replace('_', ' ')}](figures/{path.name})", ""]
    lines += ["## Reproducibility", "",
              "See episodes.csv, summary.csv, traces/, controller_summary.csv, age_ablation_paired.csv,",
              "../validation_comparison.csv, model metadata, and tensorboard/agents. No unfavorable episode is removed.", ""]
    path = directory / "RESULTS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

