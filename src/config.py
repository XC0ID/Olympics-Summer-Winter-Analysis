import yaml
from pathlib import Path

DEFAULT = Path(__file__).parent.parent / "config" / "config.yaml"

def load_config(path=None):
    with open(path or DEFAULT) as f:
        return yaml.safe_load(f)

CONFIG = load_config()