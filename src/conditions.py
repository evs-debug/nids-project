import numpy as np

from data import NUMERIC

# features that count packets or bytes seen, so a lossy tap under-reports them
VOLUME = ["src_bytes", "dst_bytes", "count", "srv_count", "dst_host_count", "dst_host_srv_count"]


def continuous_stats(X_train):
    binary = [c for c in NUMERIC if set(X_train[c].unique()) <= {0, 1}]
    continuous = [c for c in NUMERIC if c not in binary]
    return X_train[continuous].std().to_dict()


def gaussian_noise(X, sigma, stats, rng):
    out = X.copy()
    for col, std in stats.items():
        noisy = X[col].to_numpy(dtype=float) + rng.normal(0, sigma * std, len(X))
        upper = 1 if col.endswith("_rate") else None
        out[col] = np.clip(noisy, 0, upper)
    return out


def packet_loss(X, loss, rng):
    out = X.copy()
    for col in VOLUME:
        out[col] = rng.binomial(X[col].to_numpy(dtype=np.int64), 1 - loss)
    return out


def feature_dropout(X, p, rng):
    out = X.copy()
    values = X[NUMERIC].to_numpy(dtype=float)
    values[rng.random(values.shape) < p] = 0
    out[NUMERIC] = values
    return out


def resample_attack_ratio(X, y, ratio, rng):
    y = np.asarray(y)
    attacks = np.flatnonzero(y != 0)
    normals = np.flatnonzero(y == 0)
    attacks_needed = int(round(ratio / (1 - ratio) * len(normals)))
    if attacks_needed <= len(attacks):
        attacks = rng.choice(attacks, attacks_needed, replace=False)
    else:
        normals_needed = int(round(len(attacks) * (1 - ratio) / ratio))
        normals = rng.choice(normals, normals_needed, replace=False)
    idx = np.sort(np.concatenate([attacks, normals]))
    return X.iloc[idx], y[idx]
