from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "MANIFEST.csv"
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", "derived", "data_jhtdb"}


def archived_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == MANIFEST:
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() in {".pyc", ".pyo"}:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def main() -> None:
    files = archived_files()
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sha256", "bytes", "path"])
        writer.writeheader()
        for path in files:
            data = path.read_bytes()
            writer.writerow(
                {
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                    "path": path.relative_to(ROOT).as_posix(),
                }
            )
    print(f"Wrote {MANIFEST.name} for {len(files)} archived files.")


if __name__ == "__main__":
    main()
