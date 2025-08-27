# Loan Approval Prediction 

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import sklearn

# -------------------------------
# 1. Load Dataset
# -------------------------------
print(">>> Running with:", sys.executable)
df = pd.read_csv("D:\Eleevo\Task2- loan_approval\loan_approval_dataset.csv")

# Fix: remove spaces from column names
df.columns = df.columns.str.strip()

# Also remove leading/trailing spaces inside string values
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

print("\nDataset shape:", df.shape)
print("\nMissing values per column:\n", df.isnull().sum())

# Target column
target = "loan_status"
X = df.drop(columns=[target])
y = df[target]

# -------------------------------
# 2. Handle categorical/numeric
# -------------------------------
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# OneHotEncoder compatibility fix
if sklearn.__version__ >= "1.2":
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
else:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse=True)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", onehot, categorical_features)
    ]
)

# -------------------------------
# 3. Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 4. Models and Evaluation
# -------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced")
}

results = {}

for name, clf in models.items():
    from sklearn.pipeline import Pipeline
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clf", clf)
    ])
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print(f"\n=== {name} Report ===")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    results[name] = classification_report(y_test, y_pred, output_dict=True)

# -------------------------------
# 5. Save Results
# -------------------------------
results_df = pd.DataFrame(results).transpose()
results_df.to_csv("loan_approval_results.csv", index=True)

print("\n✅ Results saved to loan_approval_results.csv")
