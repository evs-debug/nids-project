from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from baselines import RuleBaseline, build_logreg
from conditions import feature_dropout, measurement_noise, packet_loss, resample_attack_ratio
from data import load_attack_names, load_split
from metrics import evaluate
from models import build_random_forest, build_xgboost, fit_weighted

RESULTS = Path(__file__).resolve().parent.parent / "results"
SEEDS = [0, 1, 2]
# one attack type per family, removed from training for the unseen-attack test
UNSEEN = ["back", "satan", "warezclient", "buffer_overflow"]


def fit_all(X, y):
    return {
        "rules": RuleBaseline().fit(X, y),
        "logreg": build_logreg().fit(X, y),
        "random_forest": fit_weighted(build_random_forest(), X, y),
        "xgboost": fit_weighted(build_xgboost(), X, y),
    }


def score_all(models, X, y):
    return {name: evaluate(y, model.predict(X)) for name, model in models.items()}


def main():
    X_full, y_full = load_split("Train")
    X_test, y_test = load_split("Test")
    attacks = load_attack_names("Train")

    idx_a, idx_b = train_test_split(
        np.arange(len(y_full)), test_size=0.2, stratify=y_full, random_state=0
    )
    X_a, y_a = X_full.iloc[idx_a], y_full.iloc[idx_a]
    X_b, y_b = X_full.iloc[idx_b], y_full.iloc[idx_b]

    rows = []

    def record(group, condition, seed, scores):
        for model, values in scores.items():
            rows.append({"group": group, "condition": condition, "seed": seed,
                         "model": model, **values})

    print("training on 80% split", flush=True)
    models_a = fit_all(X_a, y_a)
    record("in_distribution", "clean", 0, score_all(models_a, X_b, y_b))

    conditions = {
        "noise_0.1": lambda rng: (measurement_noise(X_b, 0.1, rng), y_b),
        "noise_0.3": lambda rng: (measurement_noise(X_b, 0.3, rng), y_b),
        "packet_loss_0.2": lambda rng: (packet_loss(X_b, 0.2, rng), y_b),
        "packet_loss_0.5": lambda rng: (packet_loss(X_b, 0.5, rng), y_b),
        "dropout_0.1": lambda rng: (feature_dropout(X_b, 0.1, rng), y_b),
        "dropout_0.3": lambda rng: (feature_dropout(X_b, 0.3, rng), y_b),
        "attack_ratio_0.05": lambda rng: resample_attack_ratio(X_b, y_b, 0.05, rng),
        "attack_ratio_0.2": lambda rng: resample_attack_ratio(X_b, y_b, 0.2, rng),
        "attack_ratio_0.5": lambda rng: resample_attack_ratio(X_b, y_b, 0.5, rng),
    }
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        for name, make in conditions.items():
            X_cond, y_cond = make(rng)
            record("in_distribution", name, seed, score_all(models_a, X_cond, y_cond))
        print(f"seed {seed} done", flush=True)

    print("training on full train for the official test split", flush=True)
    models_full = fit_all(X_full, y_full)
    record("stress", "official_test", 0, score_all(models_full, X_test, y_test))

    print("training without the held-out attack types", flush=True)
    keep = ~attacks.iloc[idx_a].isin(UNSEEN).to_numpy()
    models_unseen = fit_all(X_a[keep], y_a[keep])
    held_out = attacks.iloc[idx_b].isin(UNSEEN).to_numpy()
    y_b_arr = y_b.to_numpy()
    unseen_scores = {}
    for name, model in models_unseen.items():
        pred = model.predict(X_b)
        unseen_scores[name] = {
            **evaluate(y_b, pred),
            "unseen_detection": (pred[held_out] != 0).mean(),
            "unseen_family_acc": (pred[held_out] == y_b_arr[held_out]).mean(),
        }
    record("stress", "unseen_attacks", 0, unseen_scores)
    print(f"held-out attack rows in evaluation set: {held_out.sum()}", flush=True)

    raw = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    raw.to_csv(RESULTS / "experiments_raw.csv", index=False)

    agg = raw.drop(columns="seed").groupby(["group", "condition", "model"], sort=False).agg(["mean", "std"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg.round(4).to_csv(RESULTS / "experiments.csv")

    for metric in ["macro_f1", "detection_rate", "fpr"]:
        print(f"\n{metric}")
        print(raw.pivot_table(index="condition", columns="model", values=metric,
                              aggfunc="mean", sort=False).round(3).to_string())


if __name__ == "__main__":
    main()
