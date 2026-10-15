import time
from pathlib import Path

import joblib
import pandas as pd

from data import load_split
from metrics import evaluate
from models import build_random_forest, build_xgboost, fit_weighted

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MODEL_DIR = ROOT / "models"


def main():
    X_train, y_train = load_split("Train")
    X_test, y_test = load_split("Test")

    specs = {
        "xgboost": (build_xgboost, 0.5),
        "xgboost_unweighted": (build_xgboost, 0.0),
        "random_forest": (build_random_forest, 0.5),
    }
    MODEL_DIR.mkdir(exist_ok=True)
    rows = {}
    for name, (build, power) in specs.items():
        start = time.perf_counter()
        model = fit_weighted(build(), X_train, y_train, power)
        fit_s = time.perf_counter() - start

        start = time.perf_counter()
        pred = model.predict(X_test)
        ms_per_1k = (time.perf_counter() - start) / len(X_test) * 1e6

        rows[name] = {**evaluate(y_test, pred), "fit_s": fit_s, "ms_per_1k_flows": ms_per_1k}
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")

    table = pd.DataFrame(rows).T.round(4)
    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / "models.csv")

    baselines = pd.read_csv(RESULTS / "baselines.csv", index_col=0)
    print(pd.concat([baselines, table]).to_string())


if __name__ == "__main__":
    main()
