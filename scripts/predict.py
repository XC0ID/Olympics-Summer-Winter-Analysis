import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
from src.data.loader      import load_all
from src.data.cleaner     import clean_medals, clean_countries
from src.features.builder import build_features, get_feature_cols

# fix path
BASE = Path(__file__).parent.parent

def predict(country: str, year: int):
    medals, countries = load_all(config_path=str(BASE / "config/config.yaml"))
    medals    = clean_medals(medals)
    countries = clean_countries(countries)
    features  = build_features(medals, countries)
    feat_cols = get_feature_cols(features)
    model     = joblib.load(BASE / "models/trained/regression_rf.pkl")

    row = features[(features["Country"] == country) & (features["Year"] == year)]
    X   = row[feat_cols].values[:1] if not row.empty else \
          features[features["Country"] == country][feat_cols].mean().values.reshape(1, -1)

    pred = model.predict(X)[0]
    print(f"Predicted medals for {country} in {year}: {pred:.1f}")
    return pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", default="United States")
    parser.add_argument("--year",    type=int, default=2028)
    args = parser.parse_args()
    predict(args.country, args.year)