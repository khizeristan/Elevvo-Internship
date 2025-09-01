# forest_cover_classification.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# =============================
# Config
# =============================
DATA_PATH = r"D:\Eleevo\Task3- Forest\covtype.csv"
PLOT_DIR = "plots"
RANDOM_STATE = 42
os.makedirs(PLOT_DIR, exist_ok=True)

# =============================
# Load Dataset
# =============================
print("Loading data...")
data = pd.read_csv(DATA_PATH)

# Shift labels 1–7 → 0–6 (fix for XGBoost)
y = data["Cover_Type"] - 1
X = data.drop(columns=["Cover_Type"])

print(f"Data shape: X={X.shape}, y={y.shape}")
print("Class distribution:")
print(y.value_counts())

# =============================
# Train/Test Split
# =============================
print("\nPreprocessing & split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# =============================
# Helper: Train & Evaluate
# =============================
def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name, tune=False, fast=True):
    print(f"\nTraining {model_name}...")

    if tune:
        if "RandomForest" in model_name:
            param_dist = {
                "n_estimators": [50, 100],
                "max_depth": [10, None],
                "min_samples_split": [2],
            }
            clf = RandomizedSearchCV(
                RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=2),
                param_distributions=param_dist,
                n_iter=5 if fast else 10,
                cv=2 if fast else 3,
                verbose=1,
                n_jobs=2,
            )
            model = clf
        elif "XGBoost" in model_name:
            param_dist = {
                "n_estimators": [50, 100],
                "max_depth": [3, 6],
                "learning_rate": [0.1, 0.05],
            }
            clf = RandomizedSearchCV(
                XGBClassifier(
                    random_state=RANDOM_STATE,
                    use_label_encoder=False,
                    eval_metric="mlogloss",
                    n_jobs=2,
                ),
                param_distributions=param_dist,
                n_iter=5 if fast else 10,
                cv=2 if fast else 3,
                verbose=1,
                n_jobs=2,
            )
            model = clf

    # Fit model
    model.fit(X_train, y_train)

    # Best params if tuned
    if tune:
        print(f"Best parameters for {model_name}: {model.best_params_}")
        model = model.best_estimator_

    # Evaluate
    y_pred = model.predict(X_test)
    print(f"\nClassification report for {model_name}:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="viridis", xticks_rotation="vertical")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.savefig(os.path.join(PLOT_DIR, f"confusion_{model_name}.png"))
    plt.close()

    # Feature importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(indices)), importances[indices], align="center")
        plt.xticks(range(len(indices)), indices)
        plt.title(f"{model_name} - Top 20 Feature Importances")
        plt.savefig(os.path.join(PLOT_DIR, f"feature_importance_{model_name}.png"))
        plt.close()

# =============================
# Main
# =============================
def main(fast=True):
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=2)
    xgb = XGBClassifier(
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric="mlogloss",
        n_jobs=2,
    )

    # Train & evaluate both models
    train_and_evaluate(rf, X_train, X_test, y_train, y_test, "RandomForest", tune=True, fast=fast)
    train_and_evaluate(xgb, X_train, X_test, y_train, y_test, "XGBoost", tune=True, fast=fast)

if __name__ == "__main__":
    # Start in FAST mode (quick test)
    main(fast=True)

    # Change to: main(fast=False) for full hyperparameter search
