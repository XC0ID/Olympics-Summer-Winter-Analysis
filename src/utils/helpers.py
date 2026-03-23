import json
import pandas as pd
from pathlib import Path

def save_json(data: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {path}")

def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def save_csv(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")