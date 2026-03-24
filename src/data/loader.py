import pandas as pd
import yaml
from pathlib import Path

def load_config(config_path: str = "config/config.yaml"):
    """Load the YAML configuration file."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all(config_path: str = "config/config.yaml"):
    """
    Load Summer, Winter Olympics data + countries.
    Automatically detects the project root so it works from anywhere.
    """
    cfg = load_config(config_path)

    # === Automatically find project root (Olympics-ML-Analysis) ===
    # Start from this file (loader.py) and go up until we find the project folder
    current_file = Path(__file__).resolve()
    project_root = current_file.parent
    while project_root.name != "Olympics-ML-Analysis" and project_root.parent != project_root:
        project_root = project_root.parent

    # Build full path to raw data folder
    raw = project_root / cfg["data"]["raw_dir"]

    # Load the files
    summer = pd.read_csv(raw / cfg["data"]["summer_file"])
    summer["Season"] = "Summer"

    winter = pd.read_csv(raw / cfg["data"]["winter_file"])
    winter["Season"] = "Winter"

    countries = pd.read_csv(raw / cfg["data"]["countries_file"])

    medals = pd.concat([summer, winter], ignore_index=True)

    print(f"✅ Medals loaded: {medals.shape} | Countries: {countries.shape}")
    print(f"   Summer records : {len(summer):,}")
    print(f"   Winter records : {len(winter):,}")
    print(f"   Project root used: {project_root}")

    return medals, countries