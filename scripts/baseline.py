"""Optional baseline-only validation export; the full pipeline includes this automatically."""
import argparse
from pathlib import Path
from smart_grid.config import ROOT, load_config
from smart_grid.evaluation import validation_baselines

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Select an empty directory for standalone baseline references.")
    args.output.mkdir(parents=True, exist_ok=True)
    validation_baselines(load_config(args.config), args.output, args.data_dir)

