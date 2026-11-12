# Fraud Detection on Highly Imbalanced Transactions

Project 2 of the DecodeLabs Data Science internship (Batch 2026). The task: classify credit card transactions as fraud or legitimate when only about 0.17% of them are fraud, and do it without fooling myself with accuracy or leaking test information into training.

## Why accuracy is thrown out
A model that says "legitimate" for every transaction is right about 99.8% of the time and catches zero fraud. So I report precision, recall, F1, ROC-AUC and PR-AUC, plus the raw confusion matrix. I included a do-nothing baseline in the script to show this directly.

## How the pipeline avoids leakage
1. Stratified 80/20 train/test split happens first, so both sets keep the same fraud rate. The test set is never resampled or scaled.
2. Scaling and SMOTE live inside an `imblearn` Pipeline, so they are fitted only on the training part of each CV fold. (sklearn's own Pipeline can't resample.)
3. Logistic Regression gets a `StandardScaler` (regularisation is sensitive to scale, and `Amount`/`Time` dwarf the V features). Random Forest doesn't need one.
4. Hyperparameters are tuned with `GridSearchCV` (5-fold stratified), including SMOTE's `k_neighbors`, selected by ROC-AUC.
5. The best model is scored once on the untouched test set.

| Model | Grid |
|---|---|
| Logistic Regression | `C` in 0.01, 0.1, 1.0; SMOTE `k_neighbors` in 3, 5, 7 |
| Random Forest | `max_depth` in 10, 20, None; SMOTE `k_neighbors` in 3, 5 |

## Run it
```bash
pip install -r requirements.txt
# download creditcard.csv into data/
python project2_fraud_pipeline.py --fast
```
`--fast` tunes on a 30k stratified subsample of the training set and then refits the winning settings on the full training set. Without it, tuning uses all ~228k training rows and takes a long time.

Outputs go to `output/`: `test_results.csv`, `best_params.json`, `project2_evaluation.png`.

## Results
_Fill this in from `output/test_results.csv` after running on the real data._

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | | | | | |
| Random Forest | | | | | |

**What I noticed:** _write your own observations here (precision/recall trade-off, which model won and why, what the false positives cost)._

## Limitations
- ROC-AUC looks flattering on imbalanced data, which is why PR-AUC is reported too.
- A 0.5 probability threshold is used. In a real system the threshold would be set from the cost of a missed fraud versus a wrongly declined customer.
- SMOTE interpolates between existing frauds, so it can't invent fraud patterns the data has never shown.
