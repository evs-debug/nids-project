from pathlib import Path

import pandas as pd

from baselines import RuleBaseline, build_logreg
from data import load_split
from metrics import evaluate

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main():
    X_train, y_train = load_split("Train")
    X_test, y_test = load_split("Test")

    models = {"rules": RuleBaseline(), "logreg": build_logreg()}
    rows = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        rows[name] = evaluate(y_test, model.predict(X_test))

    table = pd.DataFrame(rows).T.round(4)
    print(table.to_string())
    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / "baselines.csv")


if __name__ == "__main__":
    main()
