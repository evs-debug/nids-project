import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

from data import make_preprocessor


def balanced_weights(y, power=0.5):
    # full balancing (power=1) swamps the common classes with 52 U2R rows, so soften it
    y = np.asarray(y)
    counts = np.bincount(y)
    per_class = (len(y) / (len(counts) * counts)) ** power
    return per_class[y]


def build_xgboost(seed=0):
    return make_pipeline(
        make_preprocessor(),
        XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=seed,
        ),
    )


def build_random_forest(seed=0):
    return make_pipeline(
        make_preprocessor(),
        RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed),
    )


def fit_weighted(pipe, X, y, power=0.5):
    step = pipe.steps[-1][0]
    pipe.fit(X, y, **{f"{step}__sample_weight": balanced_weights(y, power)})
    return pipe
