"""Fetch original UCI HAR archive and safely extract it. Requires internet access."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from sensor_classifier.data.loading import UCI_URL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    destination = args.root / "data"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "UCI HAR Dataset" / "train" / "X_train.txt").exists():
        print("Dataset already exists; no download attempted")
        return
    with tempfile.TemporaryDirectory(dir=destination) as tmp:
        archive = Path(tmp) / "dataset.zip"
        print(f"Downloading official dataset from {UCI_URL}")
        try:
            with urllib.request.urlopen(UCI_URL, timeout=90) as source, archive.open("wb") as target:
                shutil.copyfileobj(source, target)
        except Exception as exc:
            raise RuntimeError("Download failed; manually obtain UCI dataset 240 and unzip in data/") from exc
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        with zipfile.ZipFile(archive) as zip_file:
            for member in zip_file.infolist():
                out = (destination / member.filename).resolve()
                if not out.is_relative_to(destination.resolve()):
                    raise ValueError("Unsafe archive path")
            zip_file.extractall(destination)
        print(f"Archive SHA256: {digest}; extracted into {destination}")


if __name__ == "__main__":
    main()
