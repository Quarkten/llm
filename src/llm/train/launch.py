# Copyright 2025
# Apache-2.0 License

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from ..config import RootConfig, load_config_file, ModelConfig, TrainConfig
from .trainer import run_training


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Launch training from YAML/JSON config")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML/JSON root config")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg: RootConfig = load_config_file(args.config)

    if cfg.train is None:
        raise SystemExit("Config missing 'train' section")

    # Build model/training configs (RootConfig already validates)
    model_cfg: ModelConfig = cfg.model
    train_cfg: TrainConfig = cfg.train

    print("== Model Config ==")
    print(model_cfg.model_dump())
    print("== Train Config ==")
    print(train_cfg.model_dump())

    run_training(model_cfg, train_cfg)


if __name__ == "__main__":
    main()