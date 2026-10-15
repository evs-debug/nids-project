# AI-Based Network Intrusion Detection with Explainable Attack Classification

Computer Networks course project. Trains an ML-based intrusion detector on NSL-KDD, compares it against conventional baselines under several controlled traffic conditions, and explains its attack classifications with SHAP.

## Setup

```
pip install -r requirements.txt
python src/get_data.py
```

## Running it

Each script is independent and run from the project root.

| script | what it does | rough time |
|---|---|---|
| `src/run_baselines.py` | rule-based and logistic regression baselines on the official test split | 40 s |
| `src/run_models.py` | XGBoost and Random Forest on the official test split | 2 min |
| `src/run_experiments.py` | all four models under every controlled condition (3 seeds) plus the two stress tests | 7 min |
| `src/explain.py` | SHAP plots and example alert explanations | 1 min |

## Results

`results/experiments.csv` has the mean and standard deviation per condition and model, and `results/experiments_raw.csv` has every seed. Figures are in `results/figures/`.

`results/noise_sensitivity_std_scaled.csv` keeps the output of a first noise model that was replaced because it distorted heavy-tailed features. The commit history explains why.

Network conditions are emulated on the flow features. They are not generated with a network emulator such as Mininet.
