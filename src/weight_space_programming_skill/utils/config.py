import yaml
import os

def get_setting(key: str, default=None):
    """Load settings from `weight_space_config.yaml` located at the package root.
    Returns the value for *key* or *default* if not present.
    """
    config_path = os.path.join(os.path.dirname(__file__), "..", "weight_space_config.yaml")
    if not os.path.exists(config_path):
        return default
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get(key, default)
