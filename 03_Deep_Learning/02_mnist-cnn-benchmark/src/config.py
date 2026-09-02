"""YAML experiment configuration with command-line overrides."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parser_with_config(description: str, defaults: dict[str, object]) -> argparse.ArgumentParser:
    """Load --config first, then allow all ordinary CLI values to override it."""
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="configs/default.yaml")
    known, _ = config_parser.parse_known_args()
    path = Path(known.config)
    values = {}
    if path.exists():
        values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(values, dict):
            raise ValueError(f"Config must contain a YAML mapping: {path}")
    elif known.config != "configs/default.yaml":
        raise FileNotFoundError(f"Config file not found: {path}")
    parser = argparse.ArgumentParser(description=description, parents=[config_parser])
    parser.set_defaults(**defaults)
    parser.set_defaults(**values)
    return parser
