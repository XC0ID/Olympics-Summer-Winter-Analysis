from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

def build_rf_regressor(n_estimators=200, max_depth=6, random_state=42):
    return RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=random_state, n_jobs=-1)

def build_gbr(random_state=42):
    return GradientBoostingRegressor(
        n_estimators=200, max_depth=4,
        learning_rate=0.05, random_state=random_state)

def build_ridge():
    return Pipeline([("scaler", StandardScaler()),
                     ("ridge",  Ridge(alpha=1.0))])

def cross_validate_reg(model, X, y, cv=5):
    scores = cross_val_score(
        model, X, y, scoring="neg_root_mean_squared_error", cv=cv)
    return {"rmse_mean": round(-scores.mean(), 4),
            "rmse_std":  round(scores.std(), 4)}