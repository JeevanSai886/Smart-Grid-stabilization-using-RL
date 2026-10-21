# Dashboard release — 200,000 timesteps

- Streamlit entry point: `dashboard.py`; Windows shortcut: `run_dashboard.ps1`.
- Read `DASHBOARD_GUIDE.md` for setup, graph definitions and commands.
- Three PPO variants each use exactly 200,000 steps (600,000 combined).
- Five focused graphs: reward convergence, frequency comparison, voltage comparison,
  matched frequency response and matched voltage response.
- New time-weighted mean/RMS and peak metrics are emitted by the existing evaluation
  and validation pipeline. Frequency excludes blackout placeholders; coverage and
  blackout duration accompany comparisons. Voltage includes outage deviation.
- Original data, physics, reward, controller logic and matched scenario protocol preserved.
- TensorBoard and existing static reports remain optional complementary outputs.
- No project execution, training, evaluation, tests, dependency install or UI launch
  performed by the assistant. Only static checks were performed.
- No invented charts, trained weights or results are bundled.
