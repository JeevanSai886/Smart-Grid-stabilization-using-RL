"""Archive-only integrity check. This does not execute the packaged project."""
import argparse
from pathlib import Path
import zipfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    with zipfile.ZipFile(args.archive) as archive:
        if not archive.namelist() or archive.testzip() is not None:
            raise RuntimeError("Empty or damaged ZIP.")
        print(f"Archive readable: {len(archive.namelist())} entries.")

