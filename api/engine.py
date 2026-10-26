import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from baselines import RuleBaseline  # noqa: E402
from conditions import feature_dropout, measurement_noise, packet_loss  # noqa: E402
from data import ATTACK_TO_CLASS, CLASSES, COLUMNS, load_split  # noqa: E402
from demo import get_models  # noqa: E402
from explain import original_feature  # noqa: E402
from metrics import evaluate  # noqa: E402

FEATURES = [c for c in COLUMNS if c not in ("label", "difficulty")]
CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]

CONDITIONS = {
    "clean": ("Clean traffic", None),
    "noise_low": ("Measurement noise, sigma 0.1", lambda X, rng: measurement_noise(X, 0.1, rng)),
    "noise_high": ("Measurement noise, sigma 0.3", lambda X, rng: measurement_noise(X, 0.3, rng)),
    "loss_low": ("Packet loss 20%", lambda X, rng: packet_loss(X, 0.2, rng)),
    "loss_high": ("Packet loss 50%", lambda X, rng: packet_loss(X, 0.5, rng)),
    "dropout_low": ("Feature dropout 10%", lambda X, rng: feature_dropout(X, 0.1, rng)),
    "dropout_high": ("Feature dropout 30%", lambda X, rng: feature_dropout(X, 0.3, rng)),
}
MODEL_NAMES = ["rules", "logreg", "xgboost"]
TOP_REASONS = 5


def to_class_index(labels):
    out = []
    for raw in labels:
        name = str(raw).strip()
        if name in CLASSES:
            out.append(CLASSES.index(name))
        elif name in ATTACK_TO_CLASS:
            out.append(CLASSES.index(ATTACK_TO_CLASS[name]))
        else:
            raise ValueError(f"unknown label '{name}'")
    return np.array(out)


class Engine:
    def __init__(self):
        X, y = load_split("Train")
        idx_a, idx_b = train_test_split(
            np.arange(len(y)), test_size=0.2, stratify=y, random_state=0
        )
        trained = get_models(X.iloc[idx_a], y.iloc[idx_a])
        self.models = {"rules": RuleBaseline(), **trained}
        self.holdout_X = X.iloc[idx_b].reset_index(drop=True)
        self.holdout_y = y.iloc[idx_b].to_numpy()

        xgb = self.models["xgboost"]
        self.pre, self.clf = xgb[:-1], xgb[-1]
        self.explainer = shap.TreeExplainer(self.clf)
        sources = [original_feature(n) for n in self.pre.get_feature_names_out()]
        self.group_names = sorted(set(sources))
        self.group_cols = [
            np.flatnonzero([s == g for s in sources]) for g in self.group_names
        ]
        self._lock = threading.Lock()
        self._bench = {}

    def classify(self, flows):
        flows = flows[FEATURES].reset_index(drop=True)
        with self._lock:
            encoded = self.pre.transform(flows)
            proba = self.clf.predict_proba(encoded)
            values = self.explainer.shap_values(encoded)
            rules = self.models["rules"].predict(flows)
            logreg = self.models["logreg"].predict(flows)
        values = np.stack(values, axis=-1) if isinstance(values, list) else values
        grouped = np.stack([values[:, cols, :].sum(axis=1) for cols in self.group_cols], axis=1)
        verdict = proba.argmax(axis=1)
        own = grouped[np.arange(len(flows)), :, verdict]

        rows = []
        for i in range(len(flows)):
            top = np.argsort(-np.abs(own[i]))[:TOP_REASONS]
            reasons = []
            for g in top:
                feat = self.group_names[g]
                raw = flows.iloc[i][feat]
                value = raw if isinstance(raw, str) else float(raw)
                reasons.append({"feature": feat, "value": value, "shap": round(float(own[i, g]), 4)})
            rows.append({
                "rules": CLASSES[rules[i]],
                "logreg": CLASSES[logreg[i]],
                "xgboost": CLASSES[verdict[i]],
                "confidence": round(float(proba[i, verdict[i]]), 4),
                "proba": {c: round(float(proba[i, k]), 4) for k, c in enumerate(CLASSES)},
                "reasons": reasons,
            })
        return rows

    def draw(self, n, condition, attack_ratio, seed):
        rng = np.random.default_rng(seed)
        if attack_ratio is None:
            idx = rng.choice(len(self.holdout_y), n, replace=False)
        else:
            k = int(rng.binomial(n, attack_ratio))
            attacks = np.flatnonzero(self.holdout_y > 0)
            normals = np.flatnonzero(self.holdout_y == 0)
            idx = np.concatenate([
                rng.choice(attacks, k, replace=False),
                rng.choice(normals, n - k, replace=False),
            ])
            rng.shuffle(idx)
        flows = self.holdout_X.iloc[idx].reset_index(drop=True)
        truth = [CLASSES[t] for t in self.holdout_y[idx]]
        degrade = CONDITIONS[condition][1]
        if degrade is not None:
            flows = degrade(flows, rng)
        return flows, truth

    def benchmark(self, condition):
        if condition in self._bench:
            return self._bench[condition]
        degrade = CONDITIONS[condition][1]
        rng = np.random.default_rng(0)
        X_eval = self.holdout_X if degrade is None else degrade(self.holdout_X, rng)
        result = {}
        for name in MODEL_NAMES:
            scores = evaluate(self.holdout_y, self.models[name].predict(X_eval))
            result[name] = {
                k: round(float(scores[k]), 4)
                for k in ("macro_f1", "detection_rate", "fpr", "alert_precision")
            }
        self._bench[condition] = result
        return result

    def sample_frame(self, n, seed):
        flows, truth = self.draw(n, "clean", 0.5, seed)
        flows = flows.copy()
        flows["label"] = truth
        return flows


def read_uploaded(file_like):
    df = pd.read_csv(file_like)
    if "duration" not in df.columns:
        file_like.seek(0)
        width = pd.read_csv(file_like, header=None, nrows=1).shape[1]
        if width not in (41, 42, 43):
            raise ValueError("expected a header row, or an NSL-KDD file with 41 to 43 columns")
        file_like.seek(0)
        df = pd.read_csv(file_like, header=None, names=COLUMNS[:width])
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    truth = None
    if "label" in df.columns:
        truth = [CLASSES[i] for i in to_class_index(df["label"])]
    flows = df[FEATURES].copy()
    for col in FEATURES:
        if col not in CATEGORICAL_FEATURES:
            flows[col] = pd.to_numeric(flows[col], errors="raise")
    return flows, truth
