import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split

MODEL_DIR = Path("models/trained")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_and_save(model, X, y, name: str, test_size=0.2, random_state=42):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    model.fit(X_tr, y_tr)
    path = MODEL_DIR / f"{name}.pkl"
    joblib.dump(model, path)
    print(f"Saved: {path}")
    return model, X_te, y_te