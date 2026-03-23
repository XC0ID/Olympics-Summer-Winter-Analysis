import pandas as pd
import yaml
from pathlib import Path

def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def load_all(config_path="config/config.yaml"):
    cfg = load_config(config_path)
    raw = Path(cfg["data"]["raw_dir"])

    summer = pd.read_csv(raw / cfg["data"]["summer_file"])
    summer["Season"] = "Summer"

    winter = pd.read_csv(raw / cfg["data"]["winter_file"])
    winter["Season"] = "Winter"

    countries = pd.read_csv(raw / cfg["data"]["countries_file"])

    medals = pd.concat([summer, winter], ignore_index=True)
    print(f"Medals loaded: {medals.shape} | Countries: {countries.shape}")
    return medals, countries