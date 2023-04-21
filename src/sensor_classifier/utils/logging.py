"""Console and file logging."""
import logging
from pathlib import Path


def configure_logging(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(), logging.FileHandler(folder / "benchmark.log", encoding="utf-8")]
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=handlers, force=True)
