import os
import sys
import numpy as np
import random
import optuna
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, precision_recall_curve
)
import optuna.logging

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

if len(sys.argv) < 2:
    print("Usage: python train_KNN.py <sigma>")
    sys.exit(1)

try:
    sigma = float(sys.argv[1])
except ValueError:
    print("Error: sigma must be a float.")
    sys.exit(1)

FEATURE_PATH = os.path.join("rms", f"rms_sigma_{sigma:.5f}")
RESULT_CSV = os.path.join("results", "knn", f"results_knn_sigma_{sigma:.5f}.csv")
os.makedirs(os.path.dirname(RESULT_CSV), exist_ok=True)

optuna.logging.set_verbosity(optuna.logging.CRITICAL)

def load_data():
    X, y = [], []
    for label in [0, 1]:
        folder = os.path.join(FEATURE_PATH, str(label))
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            arr = np.load(path).flatten()
            X.append(arr)
            y.append(label)
    return np.array(X), np.array(y)

results = []

print(f"Training KNN for sigma = {sigma:.5f} ...")

for run in range(50):
    X, y = load_data()
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=SEED + run)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, stratify=y_temp, random_state=SEED + run)

    def objective(trial):
        n_neighbors = trial.suggest_int("n_neighbors", 1, 20)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        metric = trial.suggest_categorical("metric", ["euclidean", "manhattan", "chebyshev"])

        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric
        )

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + run)
        f1_scores = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val_inner = X_train[train_idx], X_train[val_idx]
            y_tr, y_val_inner = y_train[train_idx], y_train[val_idx]

            model.fit(X_tr, y_tr)
            y_pred_proba = model.predict_proba(X_val_inner)[:, 1]
            precision_vals, recall_vals, thresholds = precision_recall_curve(y_val_inner, y_pred_proba)
            f1_vals = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)
            optimal_idx = np.argmax(f1_vals)
            optimal_threshold = thresholds[optimal_idx] if thresholds.size > 0 else 0.5
            y_pred = (y_pred_proba >= optimal_threshold).astype(int)
            f1_scores.append(f1_score(y_val_inner, y_pred))

        return np.mean(f1_scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=False)

    best_params = study.best_trial.params

    final_model = KNeighborsClassifier(
    n_neighbors=best_params["n_neighbors"],
    weights=best_params["weights"],
    metric=best_params["metric"]
    )

    X_final_train = np.concatenate([X_train, X_val], axis=0)
    y_final_train = np.concatenate([y_train, y_val], axis=0)

    final_model.fit(X_final_train, y_final_train)

    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_val, final_model.predict_proba(X_val)[:, 1])
    f1_vals = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)
    optimal_idx = np.argmax(f1_vals)
    optimal_threshold = thresholds[optimal_idx] if thresholds.size > 0 else 0.5

    y_test_pred = (y_test_proba >= optimal_threshold).astype(int)

    auc = roc_auc_score(y_test, y_test_proba)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    cm = confusion_matrix(y_test, y_test_pred)
    p_misclass = cm[1, 0] / cm[1].sum() * 100 if cm[1].sum() != 0 else 0
    n_misclass = cm[0, 1] / cm[0].sum() * 100 if cm[0].sum() != 0 else 0

    results.append({
        "run": run + 1,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "p_misclass": p_misclass,
        "n_misclass": n_misclass
    })

    print(f"Run {run+1}/50 - AUC: {auc:.4f}, F1: {f1:.4f}")

df = pd.DataFrame(results)
df.to_csv(RESULT_CSV, index=False)
print(f"\nAll results saved in {RESULT_CSV}")