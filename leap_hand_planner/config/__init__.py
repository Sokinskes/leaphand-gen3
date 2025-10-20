"""Configuration management utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "leap_hand_planner/config/default.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_config_value(config: Dict[str, Any], key_path: str, default=None):
    """
    Get nested configuration value.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., "training.batch_size")
        default: Default value if key not found

    Returns:
        Configuration value
    """
    keys = key_path.split('.')
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value