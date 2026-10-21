"""Validate a completed run without retraining or rerunning the simulator."""
import sys
from smart_grid.cli import main

if __name__ == "__main__":
    main(["verify", *sys.argv[1:]])

