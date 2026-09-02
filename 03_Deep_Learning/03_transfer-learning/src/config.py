"""YAML experiment configuration with explicit command-line overrides."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parser_with_config(description: str, defaults: dict[str, object]) -> argparse.ArgumentParser:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="configs/resnet-finetune-aug.yaml")
    known, _ = config_parser.parse_known_args()
    config_path = Path(known.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError("YAML config must contain a mapping.")
    parser = argparse.ArgumentParser(description=description, parents=[config_parser])
    parser.set_defaults(**defaults)
    parser.set_defaults(**loaded)
    return parser
