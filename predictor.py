"""A small ML model the agent can call as a tool.

Why bother? LLMs are bad at *computing* predictions but good at deciding when
one is needed. So we train a real scikit-learn model and let the agent call it.
"""
import json
import sys

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import config

MODEL_PATH = config.ROOT / "model.joblib"
METRICS_PATH = config.ROOT / "model_metrics.json"
FEATURES = ["revenue_musd", "prior_growth"]


def train_model(csv_path=None, save: bool = True):
    df = pd.read_csv(csv_path or config.DATA_DIR / "financials.csv")
    X, y = df[FEATURES], df["next_growth"]
    # Hold out 20% the model never sees, so the score reflects generalisation
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = {
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "r2_test": round(float(r2_score(y_te, pred)), 3),
        "mae_test": round(float(mean_absolute_error(y_te, pred)), 4),
    }
    if save:
        joblib.dump(model, MODEL_PATH)
        METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return model, metrics


def predict_growth(revenue_musd: float, prior_growth: float) -> dict:
    """revenue_musd: revenue in millions of USD. prior_growth: decimal (0.10 = 10%)."""
    if not MODEL_PATH.exists():
        train_model()
    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([[revenue_musd, prior_growth]], columns=FEATURES)
    value = float(model.predict(row)[0])
    return {"predicted_next_growth": round(value, 4), "as_percent": f"{value:.1%}"}


if __name__ == "__main__":
    _, m = train_model()
    print("Trained. Held-out metrics:", m)
    args = [float(a) for a in sys.argv[1:3]] or [300.0, 0.10]
    print(f"predict_growth({args[0]}, {args[1]}) ->", predict_growth(*args))
