import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
from sklearn.model_selection import train_test_split
from src.data.loader      import load_all
from src.data.cleaner     import clean_medals, clean_countries
from src.features.builder import build_features, get_feature_cols
from src.evaluation.metrics import regression_metrics

def main():
    medals, countries = load_all()
    features  = build_features(clean_medals(medals), clean_countries(countries)).dropna()
    feat_cols = get_feature_cols(features)
    X = features[feat_cols].values
    y = features["TotalMedals"].values
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = joblib.load("models/trained/regression_rf.pkl")
    mets  = regression_metrics(y_te, model.predict(X_te))
    print(json.dumps(mets, indent=2))

if __name__ == "__main__":
    main()