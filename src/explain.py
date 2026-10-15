from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split

from data import CLASSES, load_split
from models import build_xgboost, fit_weighted

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIGURES = RESULTS / "figures"
PER_CLASS = 200

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SERIES = "#e1e0d9", "#c3c2b7", "#2a78d6"


def original_feature(encoded_name):
    kind, name = encoded_name.split("__", 1)
    if kind == "num":
        return name
    for source in ("protocol_type", "service", "flag"):
        if name.startswith(source + "_"):
            return source
    return name


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0, labelsize=8)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def draw_bars(ax, series, top):
    shown = series.sort_values(ascending=False).head(top)[::-1]
    ax.barh(range(len(shown)), shown.to_numpy(), height=0.55, color=SERIES)
    ax.set_yticks(range(len(shown)))
    ax.set_yticklabels(shown.index, color=INK_2, fontsize=9)
    ax.set_ylim(-0.7, len(shown) - 0.3)
    style_axes(ax)


def main():
    X, y = load_split("Train")
    idx_a, idx_b = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=0
    )
    model = fit_weighted(build_xgboost(), X.iloc[idx_a], y.iloc[idx_a])
    pre, clf = model[:-1], model[-1]

    X_b, y_b = X.iloc[idx_b], y.iloc[idx_b].to_numpy()
    rng = np.random.default_rng(0)
    chosen = np.sort(np.concatenate([
        rng.choice(np.flatnonzero(y_b == k), min(PER_CLASS, (y_b == k).sum()), replace=False)
        for k in range(len(CLASSES))
    ]))
    sample, y_s = X_b.iloc[chosen], y_b[chosen]
    encoded = pre.transform(sample)
    print("sample size", len(sample), dict(zip(CLASSES, np.bincount(y_s))))

    explainer = shap.TreeExplainer(clf)
    values = explainer.shap_values(encoded)
    values = np.stack(values, axis=-1) if isinstance(values, list) else values

    margin = clf.get_booster().predict(xgb.DMatrix(encoded), output_margin=True)
    gap = np.abs(margin - (values.sum(axis=1) + np.asarray(explainer.expected_value))).max()
    print(f"max gap between SHAP sum and model margin: {gap:.2e}")

    # summing over one-hot columns is exact for SHAP values, so the categoricals stay readable
    sources = [original_feature(n) for n in pre.get_feature_names_out()]
    features = sorted(set(sources))
    grouped = np.stack(
        [values[:, [i for i, s in enumerate(sources) if s == f], :].sum(axis=1) for f in features],
        axis=1,
    )

    overall = pd.Series(np.abs(grouped).mean(axis=(0, 2)), index=features)
    table = pd.DataFrame({"overall": overall})
    for k, name in enumerate(CLASSES):
        table[name] = np.abs(grouped[y_s == k][:, :, k]).mean(axis=0)
    table = table.sort_values("overall", ascending=False)
    FIGURES.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(RESULTS / "shap_importance.csv", index_label="feature")

    fig, ax = plt.subplots(figsize=(7.5, 5.2), facecolor=SURFACE)
    draw_bars(ax, overall, 15)
    ax.set_xlabel("mean |SHAP|, averaged over the five classes (log-odds)", color=MUTED, fontsize=8)
    fig.text(0.02, 0.965, "Which flow features drive the model's predictions", color=INK,
             fontsize=12, fontweight="bold", va="top")
    fig.text(0.02, 0.915, f"XGBoost, {len(sample)} held-out flows, up to {PER_CLASS} per class",
             color=INK_2, fontsize=9, va="top")
    fig.subplots_adjust(left=0.30, right=0.97, top=0.86, bottom=0.11)
    fig.savefig(FIGURES / "shap_global.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), facecolor=SURFACE)
    for k, name in enumerate(CLASSES):
        ax = axes.flat[k]
        draw_bars(ax, table[name], 8)
        ax.set_title(f"{name} flows", loc="left", color=INK, fontsize=10, fontweight="bold")
        ax.set_xlabel("mean |SHAP| for this class (log-odds)", color=MUTED, fontsize=8)
    axes.flat[5].axis("off")
    axes.flat[5].text(0, 0.95, "U2R rests on only 10 flows in the sample,\nso read its panel as indicative.",
                      color=INK_2, fontsize=9, va="top", transform=axes.flat[5].transAxes)
    fig.text(0.02, 0.975, "What pushes the model toward each class", color=INK, fontsize=13,
             fontweight="bold", va="top")
    fig.text(0.02, 0.94, "Top 8 features per class, measured on flows that truly belong to that class. "
             "Each panel has its own scale.", color=INK_2, fontsize=9, va="top")
    fig.subplots_adjust(left=0.14, right=0.98, top=0.86, bottom=0.08, wspace=0.75, hspace=0.45)
    fig.savefig(FIGURES / "shap_per_class.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    proba = clf.predict_proba(encoded)
    lines = ["# Example alert explanations", "",
             "For each class, the first flow in the explained sample that the model classified "
             "correctly. SHAP values are in log-odds toward the predicted class; positive pushes "
             "toward it.", ""]
    for k, name in enumerate(CLASSES):
        hits = np.flatnonzero((y_s == k) & (proba.argmax(axis=1) == k))
        if not len(hits):
            lines += [f"## {name}", "", "No correctly classified flow in the sample.", ""]
            continue
        i = hits[0]
        lines += [f"## {name} flow (model confidence {proba[i, k]:.3f})", "",
                  "| feature | value | SHAP |", "|---|---|---|"]
        contributions = pd.Series(grouped[i, :, k], index=features)
        for feat in contributions.abs().sort_values(ascending=False).head(5).index:
            value = sample.iloc[i][feat]
            shown = f"{value:.3g}" if not isinstance(value, str) else value
            lines.append(f"| {feat} | {shown} | {contributions[feat]:+.3f} |")
        lines.append("")
    (RESULTS / "example_explanations.md").write_text("\n".join(lines))

    print("\ntop features overall")
    print(table["overall"].head(8).round(3).to_string())
    for name in CLASSES:
        print(f"\ntop features for {name}")
        print(table[name].sort_values(ascending=False).head(5).round(3).to_string())
    print()
    print((RESULTS / "example_explanations.md").read_text())


if __name__ == "__main__":
    main()
