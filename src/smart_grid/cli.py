"""Run the existing smart-grid project with one compact, reproducible comparison."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

from smart_grid.config import ROOT, load_config, save_json, sha256
from smart_grid.controllers.registry import AGENTS, LEARNED


def parser():
    root = argparse.ArgumentParser(description="Five-controller delay-aware smart-grid project")
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("prepare", "smoke", "train", "evaluate", "run-all", "plot", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path)
        p.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
        if name == "prepare":
            p.add_argument("--source", choices=["synthetic"], default="synthetic")
        elif name == "smoke":
            p.add_argument("--steps", type=int, default=120)
            p.add_argument("--seed", type=int, default=42)
        elif name == "train":
            p.add_argument("--variant", choices=["aware", "no_age", "zero_delay"], default="aware")
            p.add_argument("--seed", type=int)
            p.add_argument("--steps", type=int)
            p.add_argument("--output", type=Path, required=True)
        elif name == "run-all":
            p.add_argument("--output", type=Path, default=ROOT / "results/final_run")
            p.add_argument("--seeds", nargs="+", type=int)
            p.add_argument("--reuse-data", action="store_true", help="Compatibility flag; existing synthetic data is always reused.")
            p.add_argument("--quick", action="store_true", help="Compatibility flag; retains the configured full training budget.")
            p.add_argument("--source", choices=["synthetic"], default="synthetic")
        elif name == "evaluate":
            p.add_argument("--models-dir", type=Path, required=True)
            p.add_argument("--output", type=Path, required=True)
        elif name == "plot":
            p.add_argument("--evaluation-dir", type=Path, required=True)
            p.add_argument("--models-dir", type=Path)
        elif name == "verify":
            p.add_argument("--run-dir", type=Path, default=ROOT / "results/final_run")
    return root


def check_synthetic_data(directory):
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    if provenance["actual_source"] != "synthetic":
        raise ValueError("This comparison uses the existing synthetic data/processed dataset only.")
    hashes = {}
    for split in ("train", "validation", "test"):
        file = directory / f"{split}.csv"
        hashes[split] = sha256(file)
        if hashes[split] != provenance["partitions"][split]["sha256"]:
            raise ValueError(f"{split} data changed since preprocessing. Do not mix datasets or splits.")
    return hashes


def discover_models(directory):
    models = {}
    for metadata_file in Path(directory).glob("*/metadata.json"):
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if metadata["status"] == "complete":
            controller = metadata["controller"]
            if controller in models:
                raise ValueError(f"Multiple completed models for {controller}. Use a single run.")
            models[controller] = metadata_file.parent / metadata["selected_model"]
    return models


def main(argv=None):
    args = parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "prepare":
        from smart_grid.data.preprocess import prepare_data
        prepare_data(config, source="synthetic", output_dir=args.data_dir)
        return
    if args.command == "smoke":
        from smart_grid.data.preprocess import read_profiles
        from smart_grid.envs import SmartGridEnv
        c = copy.deepcopy(config)
        c["environment"]["episode_steps"] = args.steps
        env = SmartGridEnv(read_profiles(args.data_dir / "train.csv"), c)
        env.reset(seed=args.seed)
        env.action_space.seed(args.seed)
        try:
            for _ in range(args.steps):
                _, _, terminated, truncated, _ = env.step(env.action_space.sample())
                if terminated or truncated:
                    break
            print(env.render())
        finally:
            env.close()
        return
    if args.command == "verify":
        from smart_grid.verification import verify_run
        print(verify_run(args.run_dir))
        return
    if args.command == "plot":
        from smart_grid.plotting import make_plots, write_results_summary
        make_plots(args.evaluation_dir, args.models_dir)
        print(write_results_summary(args.evaluation_dir))
        return
    hashes = check_synthetic_data(args.data_dir)
    if args.command == "train":
        from smart_grid.training import train
        if args.seed is not None:
            config["seed"] = args.seed
        if args.steps is not None and args.steps != config["training"]["total_timesteps"]:
            raise ValueError("The --steps argument must match the training budget in the chosen configuration.")
        print(train(config, args.output, args.variant, args.data_dir))
        return
    if args.command == "evaluate":
        from smart_grid.evaluation import evaluate
        # The standalone evaluator saves all five controllers and TensorBoard.
        # Use 'plot' after a full run has produced validation_comparison.csv.
        evaluate(config, discover_models(args.models_dir), args.output, args.data_dir)
        print(args.output)
        return
    if args.command == "run-all":
        from smart_grid.training import train
        from smart_grid.evaluation import validation_baselines, assemble_validation_comparison, evaluate
        from smart_grid.plotting import make_plots, write_results_summary
        from smart_grid.verification import verify_run
        if args.seeds is not None:
            if len(args.seeds) != 1:
                raise ValueError("Use one training seed per compact run; compare separate runs for a future multi-seed study.")
            config["seed"] = args.seeds[0]
        output = args.output.resolve()
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Output exists: {output}. Choose another --output folder; nothing was overwritten.")
        output.mkdir(parents=True, exist_ok=True)
        save_json(output / "run_config.json", config)
        save_json(output / "input_manifest.json", {
            "data_sha256": hashes,
            "preserved_source_sha256": {
                str(path.relative_to(ROOT)): sha256(path)
                for folder in ("envs", "grid", "controllers")
                for path in (ROOT / "src/smart_grid" / folder).glob("*.py")},
        })
        started = time.perf_counter()
        validation_baselines(config, output, args.data_dir)
        models = {}
        for controller in LEARNED:
            models[controller] = train(config, output / "models" / controller,
                                       AGENTS[controller]["variant"], args.data_dir,
                                       tensorboard_root=output / "tensorboard/agents")
        assemble_validation_comparison(output)
        evaluate(config, models, output / "evaluation", args.data_dir)
        make_plots(output / "evaluation", output / "models")
        report = write_results_summary(output / "evaluation")
        audit = verify_run(output)
        save_json(output / "completion.json", {
            "status": "complete", "requested_timesteps_per_ppo": config["training"]["total_timesteps"],
            "elapsed_seconds": time.perf_counter() - started, "results": str(report),
            "verification": str(audit), "data_source": "synthetic",
        })
        print(f"Completed. Report: {report}", flush=True)

