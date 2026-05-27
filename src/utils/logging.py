"""Shared logging utilities."""

import logging
import time
from pathlib import Path


def setup_file_logger(log_dir: Path, tag: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(log_dir / f"{tag}_{ts}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s"))
    logging.getLogger().addHandler(fh)
