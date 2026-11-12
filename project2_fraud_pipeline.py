"""
DecodeLabs Data Science - Project 2: Supervised Learning (Fraud Detection Pipeline)

Leak-free design:
  stratified split FIRST -> imblearn Pipeline (scaler -> SMOTE -> model) -> GridSearchCV
  -> final evaluation on an untouched, still-imbalanced test set.

Usage:
  python project2_fraud_pipeline.py                      # expects data/creditcard.csv
  python project2_fraud_pipeline.py --fast               # tune on a 30k stratified subsample (recommended on a laptop)
  python project2_fraud_pipeline.py --data path/to.csv --out output
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
                             average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  # NOT sklearn's Pipeline: it can't resample

SEED = 42
ROOT = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "data" / "creditcard.csv"))
ap.add_argument("--out", default=str(ROOT / "output"))
ap.add_argument("--fast", action="store_true", help="tune on a stratified subsample, then refit best params on full train set")
ap.add_argument("--tune-rows", type=int, default=30000)
args = ap.parse_args()
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 1. Load + look at the imbalance
df = pd.read_csv(args.data)
assert "Class" in df.columns, "Expected a 'Class' column (1 = fraud, 0 = legitimate)"
X, y = df.drop(columns="Class"), df["Class"]
print(f"{len(df):,} rows | fraud = {y.sum():,} ({y.mean()*100:.3f}%)")

# ---------------------------------------------------------------- 2. Stratified split BEFORE anything else
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)
print(f"train fraud rate {y_train.mean()*100:.3f}% | test fraud rate {y_test.mean()*100:.3f}% (identical by stratification)")

# ---------------------------------------------------------------- 3. Pipelines (all preprocessing lives inside)
lr_pipe = ImbPipeline([
    ("scaler", StandardScaler()),            # LR needs scaling (Amount/Time dwarf V1..V28)
    ("smote", SMOTE(random_state=SEED)),
    ("classifier", LogisticRegression(max_iter=1000, random_state=SEED)),
])
rf_pipe = ImbPipeline([                      # trees are scale-invariant -> no scaler
    ("smote", SMOTE(random_state=SEED)),
    ("classifier", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)),
])
grids = {
    "Logistic Regression": (lr_pipe, {"smote__k_neighbors": [3, 5, 7],
                                      "classifier__C": [0.01, 0.1, 1.0]}),
    "Random Forest": (rf_pipe, {"smote__k_neighbors": [3, 5],
                                "classifier__max_depth": [10, 20, None]}),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = {"roc_auc": "roc_auc", "recall": "recall", "f1": "f1", "pr_auc": "average_precision"}

# optional speed-up: tune on a subsample, but still SMOTE only inside each fold
if args.fast and len(X_train) > args.tune_rows:
    X_tune, _, y_tune, _ = train_test_split(X_train, y_train, train_size=args.tune_rows,
                                            stratify=y_train, random_state=SEED)
    print(f"--fast: tuning on {len(X_tune):,} rows ({y_tune.sum()} fraud), refitting on full training set after")
else:
    X_tune, y_tune = X_train, y_train

# ---------------------------------------------------------------- 4. Baseline (shows why accuracy is useless here)
dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
dp = dummy.predict(X_test)
print(f"\nBaseline 'always legitimate': recall={recall_score(y_test, dp):.3f} "
      f"precision={precision_score(y_test, dp, zero_division=0):.3f} "
      f"(its accuracy would look like {(dp == y_test).mean()*100:.2f}% - which is the trap)")

# ---------------------------------------------------------------- 5. Tune + evaluate on untouched test set
rows, fitted, best_params = [], {}, {}
for name, (pipe, grid) in grids.items():
    print(f"\n=== {name}: GridSearchCV over {np.prod([len(v) for v in grid.values()])} combos x 5 folds ===")
    gs = GridSearchCV(pipe, grid, scoring=scoring, refit="roc_auc", cv=cv, n_jobs=-1, verbose=0)
    gs.fit(X_tune, y_tune)
    best_params[name] = {k: (None if v is None else v) for k, v in gs.best_params_.items()}
    print("best params:", gs.best_params_, f"| CV ROC-AUC {gs.best_score_:.4f}")
    model = gs.best_estimator_ if X_tune is X_train else clone(pipe).set_params(**gs.best_params_).fit(X_train, y_train)
    fitted[name] = model

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    rows.append({"model": name,
                 "precision": precision_score(y_test, pred, zero_division=0),
                 "recall": recall_score(y_test, pred),
                 "f1": f1_score(y_test, pred),
                 "roc_auc": roc_auc_score(y_test, proba),
                 "pr_auc": average_precision_score(y_test, proba),
                 "TP": tp, "FP": fp, "FN": fn, "TN": tn})

results = pd.DataFrame(rows).set_index("model")
print("\nTEST SET RESULTS (no accuracy, by design)\n", results.round(4).to_string())
results.round(4).to_csv(OUT / "test_results.csv")
(OUT / "best_params.json").write_text(json.dumps(best_params, indent=2, default=str))

# ---------------------------------------------------------------- 6. Plots
fig, ax = plt.subplots(1, 4, figsize=(21, 4.6))
for name, m in fitted.items():
    RocCurveDisplay.from_estimator(m, X_test, y_test, name=name, ax=ax[0])
    PrecisionRecallDisplay.from_estimator(m, X_test, y_test, name=name, ax=ax[1])
ax[0].plot([0, 1], [0, 1], "k--", lw=0.8); ax[0].set_title("ROC curve (untouched test set)")
ax[1].set_title("Precision-Recall curve")
for a, (name, m) in zip(ax[2:], fitted.items()):
    ConfusionMatrixDisplay.from_estimator(m, X_test, y_test, ax=a, colorbar=False, values_format="d")
    a.set_title(f"{name}: confusion matrix")
plt.tight_layout(); plt.savefig(OUT / "project2_evaluation.png", dpi=130)
print(f"\nSaved results, params and plots to {OUT}")
