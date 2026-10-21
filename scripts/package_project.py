"""Create a portable deliverable without the machine-specific environment."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".venv", ".git", ".pytest_cache", "__pycache__", ".cache", ".verification", "deliverables"}


def main():
    destination = ROOT / "deliverables/DelayAwareSmartGrid.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(p in EXCLUDED or p.endswith(".egg-info") for p in relative.parts):
            continue
        if relative.parts[:2] == ("data", "raw") or path.suffix in {".pyc", ".part"}:
            continue
        files.append(path)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(files):
            archive.write(path, Path("DelayAwareSmartGrid") / path.relative_to(ROOT))
    with zipfile.ZipFile(destination) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"Archive failed CRC validation: {bad}")
    print(f"Packaged {len(files)} files: {destination} ({destination.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
