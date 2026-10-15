import numpy as np

from data import NUMERIC

# features that count packets or bytes seen, so a lossy tap under-reports them
VOLUME = ["src_bytes", "dst_bytes", "count", "srv_count", "dst_host_count", "dst_host_srv_count"]


# flow statistics a collector measures; content features such as num_failed_logins
# come from payload inspection and are not affected by measurement error
MEASURED = ["duration"] + VOLUME
RATES = [c for c in NUMERIC if c.endswith("_rate")]


def measurement_noise(X, sigma, rng):
    out = X.copy()
    for col in MEASURED:
        out[col] = np.clip(X[col] * (1 + rng.normal(0, sigma, len(X))), 0, None)
    for col in RATES:
        out[col] = np.clip(X[col] + rng.normal(0, sigma, len(X)), 0, 1)
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
