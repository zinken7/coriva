"""
Unified scenario config loader (single source of truth).

Loads declarative scenario configs from a directory of YAML files. Both the
CARLA runner (generate.py) and the GT builder (build_ground_truth.py) consume
these so a scenario is defined exactly once. Configs are returned ordered by
`scenario_index`.
"""

import glob
import os
from typing import Any, Dict, List

import yaml


def load_scenario_configs(scenarios_dir: str) -> List[Dict[str, Any]]:
    """Load every *.yaml scenario config from scenarios_dir, ordered by scenario_index."""
    configs: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(scenarios_dir, "*.yaml"))):
        with open(path, "r", encoding="utf-8") as f:
            configs.append(yaml.safe_load(f))
    configs.sort(key=lambda c: c.get("scenario_index", 0))
    return configs
