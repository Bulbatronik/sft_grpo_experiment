"""Checkpoint type detection utilities shared across phases."""

from __future__ import annotations

import json
from pathlib import Path


def is_lora_checkpoint(ckpt_path: str | Path) -> bool:
    """Return True if the checkpoint directory contains a LoRA adapter."""
    return (Path(ckpt_path) / "adapter_config.json").exists()


def read_adapter_config(ckpt_path: str | Path) -> dict:
    """Read and return the PEFT adapter_config.json as a dict."""
    return json.loads((Path(ckpt_path) / "adapter_config.json").read_text())
