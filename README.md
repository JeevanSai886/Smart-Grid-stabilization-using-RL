# Delay-Aware Reinforcement Learning-Based Smart Grid Stabilization System

A CPU-oriented academic project using the existing synthetic dataset, the Baran-Wu
33-bus feeder in pandapower, Gymnasium, and Stable-Baselines3 PPO.

**This release contains source code and synthetic input data. The revised code has
not been executed or tested by the assistant. No new trained models, evaluation
numbers, graphs or TensorBoard events are claimed to exist before you run it.**

## Streamlit dashboard — primary results interface

See [DASHBOARD_GUIDE.md](DASHBOARD_GUIDE.md) for the complete setup and graph definitions.
After installing the project, train/evaluate with the command below, then launch
Streamlit in another terminal:

```powershell
.\.venv\Scripts\python.exe -m smart_grid run-all --output results/final_run
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

Open http://localhost:8501. The dashboard contains reward convergence, frequency
and voltage deviation comparisons, and matched frequency/voltage response graphs.
All three PPO models use **200,000 training timesteps each**. Results populate
only after you run the experiment locally. The UI reads CSV artifacts and never
starts a simulation. TensorBoard remains available as an optional diagnostic.

## The five controllers

| Run name | Plot label | Training |
|---|---|---|
| delay_aware_agent | Aware PPO | 200,000 PPO steps; measurement age visible |
| delay_unaware_agent | No-Age PPO | 200,000 PPO steps; measurement age hidden |
| zero_delay_agent | Zero-Delay PPO | 200,000 PPO steps; instantaneous training measurements |
| rule_based | Rule-Based | No training; unchanged threshold controller |
| idle_agent | Idle | No training; battery idle and no voluntary shedding |

All five encounter the same actual delayed measurements during test scenarios.
Idle is a passive reference, **not a guaranteed worst performer** on every metric.

## What changed

- The Streamlit dashboard adds five focused interactive graphs, filters and downloads.
- New mean/RMS/peak physical deviation metrics are saved during validation and evaluation.
- Every static comparison chart overlays or groups all five controllers on the same axes,
  with a shared legend, consistent colors/markers/line styles, and baseline hatching.
- TensorBoard has five clearly named runs and identical validation/test/physical metric tags.
- All configurations use 200,000 training steps: 200 rollouts of 1,000 steps, batch size 100.
- Validation uses the same two explicit seeds after PPO updates, including the final policy.
- The three trained policies retain both their final models and their best validation checkpoints.
- Baseline and Idle do not train. Their learning-curve lines reuse measured fixed-policy references.
- The physical model, action space, reward, delay buffer and synthetic data split are preserved.
- No outcomes are hardcoded. Winners, ties, curves, summary statistics and reports are computed from your run.

## Installation and complete run (Windows PowerShell)

Use 64-bit Python 3.12. The pinned scientific stack requires Python 3.11-3.13;
Python 3.10 is not supported by this dependency set.

Open PowerShell in this extracted project folder, then run these commands in order:

    py -3.12 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe -m pip install -e . --no-deps
    .\.venv\Scripts\python.exe -m smart_grid run-all --output results/final_run

Activation is optional: these commands directly use the project's virtual environment.
The first installation needs internet access for packages. The shipped synthetic data
lets the experiment itself run offline. CPU training and AC evaluation take time;
this command performs no hyperparameter search and uses one training seed.

The final command automatically:

1. Checks the existing synthetic partition hashes; it does not switch to OPSD or regenerate data.
2. Evaluates the two non-learning baselines on the fixed validation cases.
3. Trains Aware, No-Age and Zero-Delay PPO for exactly 200,000 steps each.
4. Saves matched validation rewards and TensorBoard metrics at the configured checkpoints.
5. Evaluates all five controllers across all five requested delays and both test seeds.
6. Generates every comparison figure and the results report from actual measurements.
7. Audits saved traces, metric calculations, controller presence and TensorBoard tags.

An existing output directory is never overwritten. For another run, change the output,
for example to results/final_run_02.

You may alternatively use setup.ps1 and run_project.ps1 after allowing local scripts,
but the direct Python commands above do not require changing PowerShell execution policy.

## Optional: open TensorBoard

In another PowerShell window, from this project folder:

    .\.venv\Scripts\python.exe -m tensorboard.main --logdir results/final_run/tensorboard/agents --port 6006

Open http://localhost:6006 and select all five runs in the Scalars view:

- delay_aware_agent
- delay_unaware_agent
- zero_delay_agent
- rule_based
- idle_agent

Examples of common tags:

- validation/cumulative_reward/mean
- validation/stability_violation_duration_s/mean
- validation/blackout_duration_s/mean
- validation/blackout_events_per_hour/mean
- validation/shed_mwh/mean
- validation/battery_cycles/mean
- test_delay/fluctuation_1/stability_violation_duration_s/mean
- test_overall/stable_fraction/mean
- grid/d3_f1_seed101/observation_age_s
- grid/d3_f1_seed101/frequency_peak_deviation_hz
- grid/d3_f1_seed101/soc
- grid/d3_f1_seed101/shed_mw

Validation x-axis = PPO training timesteps. Test-delay x-axis = delay in control
steps (one second per step in the shipped configuration). Grid x-axis = episode step.
Do not compare different tag families as if their x-axis meant the same thing.

Common physical metrics exist for all five controllers. PPO optimizer losses have
no mathematical meaning for Rule-Based or Idle and are not fabricated for them;
they remain in each model's optimizer_logs/progress.csv.

## Results you will generate

After a successful run:

    results/final_run/
      run_config.json
      input_manifest.json
      completion.json
      verification.json
      baseline_validation/             measured fixed-policy episode references
      baseline_validation_curve.csv   explicitly labeled reused baseline references
      validation_comparison.csv       matched checkpoint episodes for all five
      models/
        delay_aware_agent/
        delay_unaware_agent/
        zero_delay_agent/
          metadata.json
          final_model.zip
          best_model.zip
          train.monitor.csv
          validation_episodes.csv
          validation/evaluations.npz
          checkpoints/
          optimizer_logs/progress.csv
      tensorboard/agents/
        delay_aware_agent/
        delay_unaware_agent/
        zero_delay_agent/
        rule_based/
        idle_agent/
      evaluation/
        episodes.csv
        summary.csv
        controller_summary.csv
        controller_seed_means.csv
        age_ablation_paired.csv
        age_ablation_summary.csv
        action_diagnostics.csv
        metadata.json
        trajectory_selection.json
        RESULTS.md
        traces/
        figures/

The figures are 300-dpi PNG and scalable SVG:

- reward_comparison
- reward_curve
- stability_violations
- blackout_duration
- blackout_frequency
- load_shed
- unserved_energy
- battery_cycles
- stable_fraction
- delay_robustness
- example_trajectory

All five appear on each comparison axes with a common legend. Trajectory panels
also include all five. No separate-per-agent comparison figures are generated.
Insert PNG figures at approximately 6-6.5 inches wide in Word.

## Reading the reward curve correctly

The five-controller reward curve is **validation episode reward versus PPO training
timesteps**, using the same held-out scenarios for every controller. Checkpoint means,
SD and individual observed episode points are visible; there is no smoothing.

Rule-Based and Idle have no changing policy parameters. They are evaluated once on
the same fixed validation cases, and those measured references are reused at every
checkpoint. The CSV explicitly marks this reuse. These lines are not fake training runs.

Raw PPO training episode rewards are saved in train.monitor.csv. Those stochastic
training rewards use different episodes and should not be passed off as directly
matched rewards against baselines evaluated on a different data split.

## Fair experiment

- Existing synthetic profiles: data/processed; no data/processed_opsd results are mixed in.
- Chronological train/validation/test split unchanged.
- One training seed: 42.
- Two fixed validation seeds: 10042 and 10043.
- Two matched test seeds: 101 and 202.
- 900-second test episodes.
- Communication delays: 0, 1, 3, 5 and 10 seconds.
- Renewable fluctuation scale: 1, declared before evaluation.
- 50 full test episodes; all are retained.
- Fixed-delay evaluation disables packet loss to isolate latency; training retains
  the existing stochastic congestion/loss mechanism except for Zero-Delay training.
- Select checkpoints on validation data only. The step-zero random PPO reference
  is displayed but is not selected as a trained model.
- Error bars are descriptive scenario SD. Overall bar charts average each seed's
  delay sweep before computing SD, avoiding false extra independent samples.
- There is no guarantee that Aware wins. A flat delay curve can come from a constant policy.

## Project structure

    configs/                    default and compatible 200,000-step overrides
    data/processed/             original synthetic profiles, partitions and provenance
    src/smart_grid/
      data/                     generation, cleaning, resampling and optional public-data loaders
      envs/                     Gymnasium environment and packet delay buffer
      grid/                     33-bus AC network, battery and aggregate frequency
      controllers/              unchanged baselines and shared five-controller labels
      training.py               PPO updates, checkpoints and fixed validation
      rollout.py                common seeded episode evaluator
      evaluation.py             five-controller comparisons and baseline references
      logging.py                common TensorBoard tags
      metrics.py                episode metrics from traces
      plotting.py               five-controller figures and results report
      verification.py           post-run artifact audit
      cli.py                    ordered end-to-end workflow
    scripts/                    optional task-specific command wrappers
    tests/                      physics, delay, data and comparison checks
    docs/                       architecture, methodology, data sources and protocol
    reports/                    report template, references and presentation outline
    results/                    your generated run directories
    requirements.txt            pinned runtime/test dependencies
    pyproject.toml              installable project metadata
    setup.ps1                   optional installation helper
    run_project.ps1             optional full-run helper

## Optional commands

Run tests locally:

    .\.venv\Scripts\python.exe -m pytest -q --basetemp results/test_temp_first

Use a fresh --basetemp directory on later test runs.

Re-audit the saved run without retraining or simulating:

    .\.venv\Scripts\python.exe -m smart_grid verify --run-dir results/final_run

Regenerate plots from that run without retraining:

    .\.venv\Scripts\python.exe -m smart_grid plot --evaluation-dir results/final_run/evaluation --models-dir results/final_run/models

Only if the bundled synthetic CSV files are missing, regenerate synthetic data:

    .\.venv\Scripts\python.exe -m smart_grid prepare --source synthetic

## Academic limits

The network uses balanced AC power flow with an aggregate swing/governor frequency
approximation. It is not a validated full transient grid simulator. Renewable
second-scale fluctuations, communication delays, tier allocations and component
settings contain modeling assumptions. Frequency is undefined during outages;
zero values in raw traces are placeholders paired with a blackout flag, and the
trajectory plots display gaps.

A 200,000-step run and two test scenarios support a compact, honest demonstration,
not statistical proof of convergence, real-world reliability or superiority.
Use the generated results, including unfavorable outcomes, in your report.

