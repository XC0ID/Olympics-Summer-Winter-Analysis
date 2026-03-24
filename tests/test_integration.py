import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.cleaner     import clean_medals, clean_countries
from src.features.builder import build_features, get_feature_cols
from src.models.regression import build_rf_regressor

def test_full_pipeline(sample_medals, sample_countries):
    df        = build_features(clean_medals(sample_medals),
                               clean_countries(sample_countries)).dropna()
    feat_cols = get_feature_cols(df)
    if len(df) < 5 or not feat_cols:
        return
    X     = df[feat_cols].values
    y     = df["TotalMedals"].values
    model = build_rf_regressor(n_estimators=10)
    model.fit(X, y)
    assert len(model.predict(X)) == len(y)