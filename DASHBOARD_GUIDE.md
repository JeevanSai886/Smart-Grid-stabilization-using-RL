# Smart Grid results dashboard

## Start here — Windows PowerShell

Extract the ZIP. Open PowerShell inside the `SmartGrid` folder (the folder with
`dashboard.py`, `requirements.txt` and `src`). Use 64-bit Python 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e .
```

Run training and evaluation locally:

```powershell
.\.venv\Scripts\python.exe -m smart_grid run-all --config configs/final_experiment.yaml --output results/final_run
```

Each of the **three PPO models trains for 200,000 timesteps**, sequentially:
600,000 total training environment steps, plus validation and evaluation steps.
The other two controllers are fixed baselines and do not train. Each PPO uses
200 complete rollouts of 1,000 steps. Validation and saved checkpoints remain
at 5,000-step intervals. Validation includes step zero and the final trained
policy; this gives 41 validation points per controller for a completed run.
Training may take substantial CPU time; no runtime estimate is claimed.

In a second PowerShell window, in the same project folder, launch the dashboard:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

Open **http://localhost:8501** if the browser does not open automatically.
The page can be opened before training finishes. Click **Refresh results** to
load saved validation checkpoints. Training status remains partial until each
model finishes. Test charts appear once evaluation writes `episodes.csv`.
The dashboard has no train/evaluate buttons and does not load model weights.

Optional script shortcuts, if PowerShell permits local scripts:

```powershell
.\setup.ps1
.\run_project.ps1
.\run_dashboard.ps1
```

If `results/final_run` already contains artifacts, use a fresh folder:

```powershell
.\.venv\Scripts\python.exe -m smart_grid run-all --output results/run_200k_02
.\.venv\Scripts\python.exe -m streamlit run dashboard.py -- --run-dir results/run_200k_02
```

Alternatively, type that folder in the dashboard sidebar. Folder paths containing
spaces must be quoted in command-line arguments. Existing runs are not overwritten.

Linux/macOS equivalents: use `python3 -m venv .venv`, then replace
`.\.venv\Scripts\python.exe` with `.venv/bin/python` in the commands above.
Install CPU torch and the project as shown before launching the dashboard.

## What you will see

1. **Run overview:** configured training budget, completed PPO count, selected
   controller count and completion status. Completion requires the existing
   full-pipeline artifact audit, not merely a saved model.
2. **Reward convergence:** mean validation reward versus actual training timesteps;
   three learned curves and two dashed fixed-policy references. No smoothing
   or invented scores. Baselines can span the full budget before PPO finishes.
3. **Frequency comparison:** mean absolute, RMS or peak frequency deviation, in Hz.
4. **Voltage comparison:** mean, RMS or peak worst-bus absolute deviation from
   1.0 p.u. These are deviations, not raw minimum voltage values.
5. **Matched response:** one frequency time-series graph and one worst-bus voltage
   deviation time-series graph for the same delay, fluctuation and seed for all
   selected controllers. Thresholds come from that run's configuration.

All five graphs use consistent model colours. Filters select controllers,
communication delays and fluctuation scales. The scenario selector chooses a
single matched episode for response graphs. Test filters do not change the
validation convergence graph.

## Definitions and honest interpretation

For energized samples, frequency deviation is Δf = f − nominal frequency:

- Mean absolute frequency deviation = sum(dt × |Δf|) / energized duration.
- RMS frequency deviation = sqrt(sum(dt × Δf²) / energized duration).
- Peak frequency deviation = maximum recorded within-step peak during energized steps.

During a modeled blackout, frequency is undefined. The simulator's zero placeholder
is **not** evidence of successful stabilization. New metrics store a finite zero
placeholder only when the entire episode has no energized samples; the dashboard
uses `energized_duration_s` to turn that value into missing data. Response curves
have gaps during blackouts. Fully blacked-out episodes are omitted from frequency
averages, and valid episode counts / energized time are shown in the metrics table.
Always assess blackout duration alongside frequency: a controller that trips
frequently can show deceptively small deviations while energized. Legacy frequency
metrics remain available to existing reports; these new fields have explicit semantics.

At each control step, voltage deviation is:

`dV = max(abs(Vmin - 1.0), abs(Vmax - 1.0))`.

Mean and RMS voltage deviations are time-weighted; peak is the maximum over the
episode. These are **worst-bus** metrics rather than means across all 33 buses.
Blackout voltage is included: 0 p.u. gives a deviation of 1 p.u. This keeps outage
penalties visible. The voltage reference line is drawn only for symmetric configured
voltage limits. A single absolute-deviation threshold cannot represent asymmetric
upper and lower limits faithfully.

Comparison charts first average the selected delay/fluctuation cells within each
test seed, then calculate mean and sample SD across seeds. Only matched cells are
included; duplicates are rejected. Peak bars therefore show the **mean episode
peak**, not the single worst episode across the entire run. Error bars are descriptive
scenario SD, not confidence intervals across independently trained models. With one
seed, no SD estimate is available. A 200,000-step budget does not guarantee convergence.
The project still uses one training seed and two test seeds by default.

## Downloads

- Hover over a graph and use the camera icon to save a high-resolution PNG.
- Expand **Detailed metrics and exports** for the comparison CSV and selected
  episode CSVs packaged with the filter choices and run configuration.
- Existing static report figures and TensorBoard logs remain available. The
  Streamlit page intentionally presents only the five essential graphs.

## Existing trained models / older results

The new ZIP contains source and synthetic input data, not trained models or test
results. If you have an older completed run with compatible model weights, you
can evaluate those locally without retraining, but their training budget remains
whatever their metadata records; they do not become 200,000-step models.

To refresh an older run's metrics while retaining its own training configuration:

```powershell
.\.venv\Scripts\python.exe -m smart_grid evaluate --config results/old_run/run_config.json --models-dir results/old_run/models --output results/old_run/dashboard_evaluation
```

After that succeeds, keep the original `evaluation` folder as a backup and move
`dashboard_evaluation` to `evaluation` within that older run. Select `results/old_run`
in the sidebar. The new dashboard needs the new metric columns; it will explain
when an older `episodes.csv` lacks them. Use a fresh full run for the requested
200,000-step comparison. Never mix models or results from separate runs.

## Verification status

The delivered changes received static source/syntax, configuration, preservation
and ZIP checks only. **The assistant did not execute the project, install its
runtime dependencies, train, simulate, run tests or launch Streamlit.** No runtime
or visual verification is claimed. The historical `requirements-lock.txt` is
retained as a record of the supplied simulation stack, not as a tested dashboard
lockfile; install from the current project requirements instead.
