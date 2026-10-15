import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from baselines import RuleBaseline, build_logreg
from conditions import feature_dropout, measurement_noise, packet_loss
from data import CLASSES, load_split
from explain import original_feature
from metrics import evaluate
from models import build_xgboost, fit_weighted

CACHE = Path(__file__).resolve().parent.parent / "models" / "demo.joblib"
CONDITIONS = {
    "clean": None,
    "noise": lambda X, rng: measurement_noise(X, 0.3, rng),
    "packet_loss": lambda X, rng: packet_loss(X, 0.5, rng),
    "dropout": lambda X, rng: feature_dropout(X, 0.3, rng),
}


def get_models(X_train, y_train):
    try:
        return joblib.load(CACHE)
    except Exception:
        pass
    print("training on 80% of NSL-KDD (first run only, about a minute)...", flush=True)
    models = {
        "logreg": build_logreg().fit(X_train, y_train),
        "xgboost": fit_weighted(build_xgboost(), X_train, y_train),
    }
    CACHE.parent.mkdir(exist_ok=True)
    joblib.dump(models, CACHE)
    return models


def reasons(model, flows, predicted, k=3):
    pre, clf = model[:-1], model[-1]
    values = shap.TreeExplainer(clf).shap_values(pre.transform(flows))
    values = np.stack(values, axis=-1) if isinstance(values, list) else values
    sources = [original_feature(n) for n in pre.get_feature_names_out()]
    out = []
    for i, cls in enumerate(predicted):
        contribution = pd.Series(values[i, :, cls], index=sources).groupby(level=0).sum()
        parts = []
        for feat in contribution.abs().nlargest(k).index:
            value = flows.iloc[i][feat]
            parts.append(f"{feat}={value if isinstance(value, str) else format(value, '.3g')}")
        out.append(", ".join(parts))
    return out


def main():
    parser = argparse.ArgumentParser(description="Live demo of the intrusion detector")
    parser.add_argument("--condition", choices=list(CONDITIONS), default="clean")
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    X, y = load_split("Train")
    idx_a, idx_b = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=0
    )
    models = {"rules": RuleBaseline(), **get_models(X.iloc[idx_a], y.iloc[idx_a])}
    X_b, y_b = X.iloc[idx_b], y.iloc[idx_b].to_numpy()

    rng = np.random.default_rng(args.seed)
    degrade = CONDITIONS[args.condition]
    X_eval = X_b if degrade is None else degrade(X_b, rng)

    picks = np.concatenate([
        rng.choice(np.flatnonzero(y_b == k), args.per_class, replace=False)
        for k in range(len(CLASSES))
    ])
    flows, truth = X_eval.iloc[picks], y_b[picks]
    preds = {name: model.predict(flows) for name, model in models.items()}
    proba = models["xgboost"].predict_proba(flows)

    print(f"\nconditions: {args.condition}   (flows were never seen in training)\n")
    table = pd.DataFrame({
        "true": [CLASSES[t] for t in truth],
        "rules": [CLASSES[p] for p in preds["rules"]],
        "logreg": [CLASSES[p] for p in preds["logreg"]],
        "xgboost": [f"{CLASSES[p]} ({proba[i, p]:.2f})" for i, p in enumerate(preds["xgboost"])],
        "why xgboost says so": reasons(models["xgboost"], flows, preds["xgboost"]),
    })
    print(table.to_string(index=False))
    correct = {name: int((p == truth).sum()) for name, p in preds.items()}
    print("\ncorrect out of", len(truth), "->", ", ".join(f"{n} {c}" for n, c in correct.items()))

    scores = {name: evaluate(y_b, model.predict(X_eval)) for name, model in models.items()}
    bench = pd.DataFrame(scores).T[["macro_f1", "detection_rate", "fpr", "alert_precision"]].round(3)
    print(f"\nwhole held-out set ({len(X_eval)} flows), same conditions")
    print(bench.to_string())


if __name__ == "__main__":
    main()
