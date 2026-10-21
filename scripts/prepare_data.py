import sys
from smart_grid.cli import main

if __name__ == "__main__":
    main(["prepare", *sys.argv[1:]])

