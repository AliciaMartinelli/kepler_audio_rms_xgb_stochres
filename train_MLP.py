import os
import sys
import numpy as np
import random
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    precision_recall_curve, confusion_matrix
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

if len(sys.argv) < 2:
    print("Usage: python train_MLP.py <sigma>")
    sys.exit(1)

try:
    sigma = float(sys.argv[1])
except ValueError:
    print("Error: sigma must be a float.")
    sys.exit(1)

FEATURE_PATH = os.path.join("rms", f"rms_sigma_{sigma:.5f}")
RESULT_CSV = os.path.join("results", "mlp", f"results_mlp_sigma_{sigma:.5f}.csv")
os.makedirs(os.path.dirname(RESULT_CSV), exist_ok=True)

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

def build_model(input_dim):
    model = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
    return model

results = []

print(f"Training MLP for sigma = {sigma:.5f} ...")

for run in range(50):
    X, y = load_data()
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=SEED + run)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, stratify=y_temp, random_state=SEED + run)

    class_weights_array = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = {i: w for i, w in zip(np.unique(y_train), class_weights_array)}

    model = build_model(X_train.shape[1])
    es = EarlyStopping(patience=5, restore_best_weights=True)

    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=50,
              batch_size=32,
              class_weight=class_weights,
              verbose=0,
              callbacks=[es])

    y_val_proba = model.predict(X_val, verbose=0).flatten()
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_val, y_val_proba)
    f1_vals = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)
    optimal_idx = np.argmax(f1_vals)
    optimal_threshold = thresholds[optimal_idx] if thresholds.size > 0 else 0.5

    y_test_proba = model.predict(X_test, verbose=0).flatten()
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
print(f"\nAlle Ergebnisse gespeichert in {RESULT_CSV}")
