import sys
from smart_grid.cli import main

if __name__ == "__main__":
    main(["smoke", *sys.argv[1:]])

