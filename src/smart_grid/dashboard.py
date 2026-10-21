"""Results-only dashboard; imports no simulator, training code or model weights."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from smart_grid.controllers.registry import AGENTS, ORDER, LEARNED

ROOT = Path(__file__).resolve().parents[2]
METRICS = {
    "Mean absolute": ("mean_absolute_frequency_deviation_hz", "mean_voltage_deviation_pu"),
    "RMS": ("rms_frequency_deviation_hz", "rms_voltage_deviation_pu"),
    "Peak": ("peak_energized_frequency_deviation_hz", "peak_voltage_deviation_pu"),
}


@st.cache_data(show_spinner=False)
def read_csv(filename: str, modified_ns: int, size: int) -> pd.DataFrame:
    # File identity includes mtime and size so Refresh sees newly saved checkpoints.
    return pd.read_csv(filename)


def csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        stat = path.stat()
        return read_csv(str(path), stat.st_mtime_ns, stat.st_size)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.warning(f"Cannot read {path.name} yet ({type(exc).__name__}). Refresh after the write completes.")
        return pd.DataFrame()


def metadata(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def style(fig, ylabel, xlabel=""):
    fig.update_layout(
        template="plotly_white", height=390, margin=dict(l=20, r=20, t=25, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=12, color="#172B4D"),
        xaxis_title=xlabel, yaxis_title=ylabel,
        legend=dict(orientation="h", yanchor="top", y=-0.25, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#E8EDF4", zerolinecolor="#B6C5D8")
    return fig


def line(fig, x, y, controller, reference=False):
    details = AGENTS[controller]
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", name=details["label"] + (" · fixed reference" if reference else ""),
        line=dict(color=details["color"], width=2.6, dash="dash" if reference else "solid"),
        connectgaps=False,
    ))


def show(fig, key):
    st.plotly_chart(fig, use_container_width=True, key=key, config={
        "displaylogo": False,
        "toImageButtonOptions": {"format": "png", "filename": key, "scale": 3},
    })


def matched(frame, controllers, cells):
    """Retain only cells with exactly one row for every selected controller."""
    data = frame[frame.controller.isin(controllers)].copy()
    if data.empty:
        return data, 0
    if data.duplicated(["controller", *cells]).any():
        st.error("Duplicate result rows detected. Choose a clean run folder; duplicate cells are not averaged.")
        return data.iloc[:0], len(data)
    counts = data.groupby(cells, dropna=False).controller.nunique().rename("_count").reset_index()
    result = data.merge(counts[counts._count == len(controllers)][cells], on=cells, how="inner")
    return result, len(data) - len(result)


def convergence(run, controllers, budget):
    frame = csv(run / "validation_comparison.csv")
    if frame.empty:
        parts = [csv(run / "baseline_validation_curve.csv")]
        parts.extend(csv(p) for p in sorted((run / "models").glob("*/validation_episodes.csv")))
        parts = [part for part in parts if not part.empty]
        frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    required = {"controller", "training_timesteps", "episode_seed", "cumulative_reward"}
    if frame.empty or not required.issubset(frame.columns):
        st.info("Convergence will appear after the first validation checkpoint. Train locally, then click Refresh results.")
        return
    frame = frame[frame.controller.isin(controllers)]
    if frame.empty:
        st.info("The selected controllers have no validation checkpoints yet.")
        return
    if frame.duplicated(["controller", "training_timesteps", "episode_seed"]).any():
        st.error("Duplicate validation rows: convergence is unavailable for this folder.")
        return
    fig = go.Figure()
    for controller in controllers:
        data = frame[frame.controller == controller]
        if data.empty:
            continue
        curve = data.groupby("training_timesteps").cumulative_reward.mean().sort_index()
        line(fig, curve.index, curve.values, controller, controller not in LEARNED)
    style(fig, "Mean validation episode reward ↑", "Training timesteps")
    fig.update_xaxes(range=[0, max(budget, int(frame.training_timesteps.max()))], tickformat=",")
    show(fig, "reward_convergence")
    st.caption("Unsmoothed mean on fixed validation seeds after PPO updates. Dashed baselines are measured once and reused; they do not train. A plateau alone does not prove convergence. Test filters below do not affect this chart.")


def comparison_chart(summary, metric, controllers, ylabel):
    fig = go.Figure()
    for controller in controllers:
        if controller not in summary.index:
            continue
        row = summary.loc[controller]
        value = row[(metric, "mean")]
        if pd.isna(value):
            continue
        sd = row[(metric, "std")]
        fig.add_trace(go.Bar(
            x=[AGENTS[controller]["label"]], y=[value], name=AGENTS[controller]["label"],
            marker_color=AGENTS[controller]["color"],
            marker_pattern_shape="/" if controller not in LEARNED else "",
            error_y=dict(type="data", array=[0 if pd.isna(sd) else sd], visible=True),
            text=[f"{value:.4g}"], textposition="outside", cliponaxis=False,
            hovertemplate="%{x}<br>Mean: %{y:.5g}<extra></extra>",
        ))
    style(fig, ylabel)
    fig.update_layout(showlegend=False, hovermode="closest")
    fig.update_yaxes(rangemode="tozero")
    return fig


def responses(run, frame, controllers, config):
    st.subheader("Matched scenario response")
    choices = sorted(set(zip(frame.delay_steps, frame.fluctuation_scale, frame.episode_seed)))
    dt = float(config.get("environment", {}).get("control_dt_s", 1))
    selected = st.selectbox("Same test scenario for every controller", choices,
                            format_func=lambda v: f"Delay {v[0] * dt:g} s · fluctuation ×{v[1]:g} · seed {v[2]}")
    delay, fluctuation, seed = selected
    freq, volt = go.Figure(), go.Figure()
    for controller in controllers:
        path = run / "evaluation/traces" / f"{controller}_d{int(delay)}_f{fluctuation:g}_seed{int(seed)}.csv"
        trace = csv(path)
        required = {"time_s", "frequency_deviation_hz", "voltage_min_pu", "voltage_max_pu", "blackout"}
        if trace.empty or not required.issubset(trace.columns):
            st.warning(f"Trace unavailable for {AGENTS[controller]['label']}.")
            continue
        alive = ~trace.blackout.astype(bool)
        f = trace.frequency_deviation_hz.where(alive)
        v = np.maximum((trace.voltage_min_pu - 1).abs(), (trace.voltage_max_pu - 1).abs())
        line(freq, trace.time_s, f, controller)
        line(volt, trace.time_s, v, controller)
    env = config.get("environment", {})
    if "frequency_limit_hz" in env:
        limit = float(env["frequency_limit_hz"])
        freq.add_hrect(y0=-limit, y1=limit, fillcolor="#008566", opacity=0.06, line_width=0)
        for y in (-limit, limit):
            freq.add_hline(y=y, line_dash="dot", line_color="#8896A8")
    # An absolute-deviation limit is exact only for symmetric voltage bounds.
    low, high = env.get("voltage_min_pu"), env.get("voltage_max_pu")
    if low is not None and high is not None and np.isclose(1-low, high-1):
        volt.add_hline(y=high-1, line_dash="dot", line_color="#8896A8",
                      annotation_text="Voltage tolerance")
    left, right = st.columns(2)
    with left:
        st.markdown("**Frequency deviation**")
        show(style(freq, "Δf (Hz)", "Episode time (s)"), "frequency_response")
    with right:
        st.markdown("**Worst-bus voltage deviation**")
        show(style(volt, "max |V − 1.0| (p.u.)", "Episode time (s)"), "voltage_response")
    st.caption("Frequency gaps mark blackout periods, when frequency is undefined. Voltage includes outages (0 p.u. → deviation 1 p.u.). Traces show one matched episode; comparison bars aggregate the selected scenarios.")


def main():
    st.set_page_config(page_title="Smart Grid | Results", page_icon="⚡", layout="wide")
    st.markdown("""<style>
    .block-container {max-width:1450px;padding-top:2rem;padding-bottom:3rem;}
    [data-testid="stMetric"] {background:white;border:1px solid #e3e9f2;border-radius:12px;padding:16px;}
    h1 {letter-spacing:-1.2px;} [data-testid="stSidebar"] {border-right:1px solid #e3e9f2;}
    </style>""", unsafe_allow_html=True)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-dir", default=str(ROOT / "results/final_run"))
    args, _ = parser.parse_known_args()
    with st.sidebar:
        st.markdown("## ⚡ Smart Grid")
        st.caption("STABILIZATION · RESULTS EXPLORER")
        run_text = st.text_input("Results folder", value=args.run_dir,
                                 help="Folder containing run_config.json, models/ and evaluation/.")
        if st.button("Refresh results", use_container_width=True):
            st.cache_data.clear()
        controllers = st.multiselect("Controllers", list(ORDER), default=list(ORDER),
                                     format_func=lambda k: AGENTS[k]["label"])
        st.divider()
        st.caption("Read-only dashboard. Opening this page never starts training or evaluation.")
    st.title("Smart grid stabilization")
    st.markdown("Learning progress and matched controller performance")
    run = Path(run_text).expanduser()
    if not run.is_absolute():
        run = ROOT / run
    if not controllers:
        st.info("Select at least one controller in the sidebar.")
        return
    config = metadata(run / "run_config.json")
    completion = metadata(run / "completion.json")
    budget = int(config.get("training", {}).get("total_timesteps", 200000))
    models = [metadata(run / "models" / c / "metadata.json") for c in LEARNED]
    completed = sum(m.get("status") == "complete" for m in models)
    cards = st.columns(4)
    cards[0].metric("Training budget / PPO", f"{budget:,}")
    cards[1].metric("PPO models completed", f"{completed} / 3")
    cards[2].metric("Selected controllers", len(controllers))
    cards[3].metric("Run status", "Complete" if completion.get("status") == "complete" else "In progress" if config else "Not started")
    st.caption("Synthetic data · Baran–Wu 33-bus feeder · 3 PPO variants + 2 fixed baselines · Best validation checkpoints used for testing")
    if not config:
        st.info("No run metadata found. Start the experiment locally or select an existing run folder. No sample results are displayed.")
        st.code(".\\.venv\\Scripts\\python.exe -m smart_grid run-all --output results/final_run", language="powershell")
    if any(m.get("status") == "interrupted" for m in models):
        st.warning("An interrupted model is present. Curves may be partial; use a new output folder for fresh training.")
    st.subheader("Reward convergence")
    convergence(run, controllers, budget)
    st.divider()
    episodes = csv(run / "evaluation/episodes.csv")
    required = {"controller", "delay_steps", "delay_seconds", "fluctuation_scale", "episode_seed",
                "duration_s", "blackout_duration_s", "stable_fraction", "cumulative_reward"}
    if episodes.empty:
        st.info("Test comparisons will appear after local evaluation has saved evaluation/episodes.csv.")
        return
    if not required.issubset(episodes.columns):
        st.error("This results file is missing required scenario columns. Run the updated evaluation pipeline into a fresh folder.")
        return
    st.subheader("Controller performance")
    a, b, c = st.columns([1, 1, 1])
    delays = a.multiselect("Communication delay (s)", sorted(episodes.delay_seconds.unique()),
                            default=sorted(episodes.delay_seconds.unique()))
    fluctuations = b.multiselect("Fluctuation scale", sorted(episodes.fluctuation_scale.unique()),
                                 default=sorted(episodes.fluctuation_scale.unique()))
    choice = c.selectbox("Deviation statistic", list(METRICS))
    fm, vm = METRICS[choice]
    if not {fm, vm, "energized_duration_s"}.issubset(episodes.columns):
        st.warning("These are older results without the new deviation metrics. Convergence remains available. Use the updated evaluator with your existing compatible models and a new output directory (see DASHBOARD_GUIDE.md), or run the full 200,000-step experiment.")
        return
    data = episodes[episodes.delay_seconds.isin(delays) & episodes.fluctuation_scale.isin(fluctuations)].copy()
    data, dropped = matched(data, controllers, ["delay_steps", "fluctuation_scale", "episode_seed"])
    if dropped:
        st.warning(f"Excluded {dropped} rows with unmatched scenario cells; only matched comparisons are shown.")
    if data.empty:
        st.info("No matched results for these filters. Select delays, fluctuations and controllers with completed results.")
        return
    # Frequency is not defined in a fully blacked-out episode. Never rank its zero placeholder.
    data.loc[data.energized_duration_s <= 0, fm] = np.nan
    metrics = [fm, vm, "blackout_duration_s", "stable_fraction", "cumulative_reward"]
    per_seed = data.groupby(["controller", "episode_seed"])[metrics].mean()
    summary = per_seed.groupby("controller").agg(["mean", "std", "count"])
    left, right = st.columns(2)
    with left:
        st.markdown(f"**{choice} frequency deviation**")
        show(comparison_chart(summary, fm, controllers, "Frequency deviation (Hz) ↓"), "frequency_comparison")
    with right:
        st.markdown(f"**{choice} worst-bus voltage deviation**")
        show(comparison_chart(summary, vm, controllers, "Voltage deviation (p.u.) ↓"), "voltage_comparison")
    st.caption("Lower deviation is better. Bars average selected delay/fluctuation cells within each test seed, then average seeds. Error bars are descriptive ±1 sample SD across test seeds, not confidence intervals across independent training runs. One seed gives no SD estimate.")
    st.caption("Frequency mean/RMS use energized samples; Peak uses energized within-step peaks. Fully blacked-out episodes have no frequency estimate and are omitted from frequency averages. Always compare blackout duration and energized coverage too; low frequency deviation alone does not establish a better controller.")
    means = summary.xs("mean", axis=1, level=1).reindex(controllers)
    counts = data.groupby("controller").size().reindex(controllers)
    valid = data.groupby("controller")[fm].count().reindex(controllers)
    coverage = data.groupby("controller").energized_duration_s.sum() / data.groupby("controller").duration_s.sum()
    table = pd.DataFrame({
        "Controller": [AGENTS[k]["label"] for k in controllers],
        "Frequency (Hz)": means[fm].to_numpy(), "Voltage (p.u.)": means[vm].to_numpy(),
        "Stable time (%)": means.stable_fraction.to_numpy() * 100,
        "Blackout / episode (s)": means.blackout_duration_s.to_numpy(),
        "Energized time (%)": coverage.reindex(controllers).to_numpy() * 100,
        "Mean reward": means.cumulative_reward.to_numpy(),
        "Matched episodes": counts.to_numpy(), "Frequency-valid episodes": valid.to_numpy(),
    })
    with st.expander("Detailed metrics and exports", expanded=False):
        st.dataframe(table.round(5), hide_index=True, use_container_width=True)
        st.download_button("Download comparison CSV", table.to_csv(index=False),
                           "controller_comparison.csv", "text/csv")
        export = io.BytesIO()
        with zipfile.ZipFile(export, "w", zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr("selected_episodes.csv", data.to_csv(index=False))
            bundle.writestr("comparison.csv", table.to_csv(index=False))
            bundle.writestr("selection.json", json.dumps({"controllers": controllers, "delay_seconds": list(map(float, delays)),
                             "fluctuation_scales": list(map(float, fluctuations)), "statistic": choice, "run_config": config}, indent=2))
        st.download_button("Download selected results ZIP", export.getvalue(), "selected_results.zip", "application/zip")
        st.caption("Use the camera icon on a graph to export a high-resolution PNG.")
    responses(run, data, controllers, config)
    with st.expander("Metric definitions and study scope"):
        st.markdown("""
- **Frequency deviation:** Δf = f − nominal frequency. Mean absolute and RMS are time-weighted on energized control-step samples. Peak uses the largest within-step absolute excursion while energized.
- **Voltage deviation:** max(|Vmin − 1|, |Vmax − 1|) across buses at each control step. Mean and RMS are time-weighted; Peak is the episode maximum. This is a worst-bus metric, not an average across buses. Outages are included.
- **Convergence:** mean cumulative reward on the fixed validation seeds; baseline reference lines are not learned curves.
- **Study scope:** one training seed, two default test seeds, synthetic inputs and an approximate frequency model. More timesteps do not guarantee convergence or superiority.
        """)
