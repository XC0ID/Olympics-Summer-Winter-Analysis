import pandas as pd

REQUIRED_MEDAL_COLS   = ["Country", "Year", "Medal", "Season"]
REQUIRED_COUNTRY_COLS = ["Country"]

def validate_medals(df: pd.DataFrame) -> bool:
    missing = [c for c in REQUIRED_MEDAL_COLS if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        return False
    print("Medals validation passed.")
    return True

def validate_countries(df: pd.DataFrame) -> bool:
    missing = [c for c in REQUIRED_COUNTRY_COLS if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        return False
    print("Countries validation passed.")
    return True