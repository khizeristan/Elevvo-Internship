#!/usr/bin/env python3
"""
Student Score Prediction Pipeline
---------------------------------
- Loads the "Student_performance_data _.csv"
- Cleans data, basic EDA
- Train/Test split
- Linear Regression (StudyHours -> GPA)
- Polynomial Regression (deg=2,3)
- Extended Linear Regression with multiple features
- Repeats modeling for GradeClass as "exam score"
- Saves plots and metrics CSVs in the working directory


"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



DEFAULT_DATA_PATH = "D:\Eleevo\Task1-Student Performance\Student_performance_data _.csv"  
OUTPUT_DIR = "."  


def ensure_path(path: str) -> str:
    
    if path and os.path.exists(path):
        return path
    if os.path.exists(DEFAULT_DATA_PATH):
        return DEFAULT_DATA_PATH
    # Fallback to a local file with same name if present
    local = os.path.join(os.getcwd(), "D:\Eleevo\Task1-Student Performance\Student_performance_data _.csv")
    if os.path.exists(local):
        return local
    raise FileNotFoundError("CSV file not found. Provide a valid path.")


def savefig(path):
  
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"[saved] {path}")


def metrics(y_true, y_hat):
    mae = mean_absolute_error(y_true, y_hat)
    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_hat)
    return mae, mse, rmse, r2


def describe_and_plot(df):
    # Summary of selected columns
    summary = df[["GPA", "GradeClass", "StudyHours", "Absences"]].describe().T
    print("\n=== Summary Statistics (Selected Columns) ===")
    print(summary)

    # Study Hours vs GPA
    plt.figure()
    plt.scatter(df["StudyHours"], df["GPA"], alpha=0.5)
    plt.xlabel("Study Hours per Week")
    plt.ylabel("GPA")
    plt.title("Study Hours vs GPA")
    savefig(os.path.join(OUTPUT_DIR, "study_hours_vs_gpa.png"))

    # Study Hours vs GradeClass
    plt.figure()
    plt.scatter(df["StudyHours"], df["GradeClass"], alpha=0.5)
    plt.xlabel("Study Hours per Week")
    plt.ylabel("GradeClass (Exam Score)")
    plt.title("Study Hours vs GradeClass")
    savefig(os.path.join(OUTPUT_DIR, "study_hours_vs_gradeclass.png"))


def simple_linear_and_poly(df, target_col: str, label: str):
    """
    Fit Linear & Polynomial models using StudyHours -> target_col.
    Returns a dict of metrics and saves plots.
    """
    X = df[["StudyHours"]].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear regression
    lin = LinearRegression().fit(X_train, y_train)
    y_pred = lin.predict(X_test)
    lin_mae, lin_mse, lin_rmse, lin_r2 = metrics(y_test, y_pred)

    # Plot predictions vs actual (only for GPA to avoid too many charts)
    if target_col == "GPA":
        plt.figure()
        plt.scatter(range(len(y_test)), y_test, label="Actual", alpha=0.7)
        plt.scatter(range(len(y_test)), y_pred, label="Predicted", alpha=0.7)
        plt.xlabel("Test Sample Index")
        plt.ylabel("GPA")
        plt.title("Linear Regression: Actual vs Predicted GPA (Test Set)")
        plt.legend()
        savefig(os.path.join(OUTPUT_DIR, "lin_actual_vs_pred.png"))

        residuals = y_test - y_pred
        plt.figure()
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(0)
        plt.xlabel("Predicted GPA")
        plt.ylabel("Residual (Actual - Predicted)")
        plt.title("Linear Regression Residuals")
        savefig(os.path.join(OUTPUT_DIR, "lin_residuals.png"))

    # Polynomial regression (deg 2 & 3)
    poly_rows = []
    x_grid = np.linspace(df["StudyHours"].min(), df["StudyHours"].max(), 200).reshape(-1, 1)

    # Figure with fits
    plt.figure()
    alpha_sc = 0.3 if len(df) > 500 else 0.5
    plt.scatter(df["StudyHours"], df[target_col], alpha=alpha_sc)

    # Linear line
    y_line = lin.predict(x_grid)
    plt.plot(x_grid, y_line, label="Linear Fit")

    for deg in [2, 3]:
        poly = PolynomialFeatures(degree=deg, include_bias=False)
        X_poly_train = poly.fit_transform(X_train)
        X_poly_test = poly.transform(X_test)
        lin_poly = LinearRegression().fit(X_poly_train, y_train)
        y_poly_pred = lin_poly.predict(X_poly_test)
        p_mae, p_mse, p_rmse, p_r2 = metrics(y_test, y_poly_pred)
        poly_rows.append({
            "Model": f"Polynomial (deg={deg})->{label}",
            "MAE": p_mae, "MSE": p_mse, "RMSE": p_rmse, "R2": p_r2
        })

        # Plot curve
        plt.plot(x_grid, lin_poly.predict(poly.transform(x_grid)), label=f"Poly Deg {deg}")

    plt.xlabel("Study Hours per Week")
    plt.ylabel(label)
    plt.title(f"Model Fits: Linear vs Polynomial → {label}")
    plt.legend()
    fname = f"fit_curves_{target_col.lower()}.png"
    savefig(os.path.join(OUTPUT_DIR, fname))

    # Return metrics rows
    rows = [{
        "Model": f"Linear (StudyHours)->{label}",
        "MAE": lin_mae, "MSE": lin_mse, "RMSE": lin_rmse, "R2": lin_r2
    }] + poly_rows

    # For convenience, return the linear model coefficients for GPA
    coef, intercept = None, None
    if target_col == "GPA":
        coef = float(lin.coef_[0])
        intercept = float(lin.intercept_)
        with open(os.path.join(OUTPUT_DIR, "simple_linear_model_gpa.txt"), "w") as f:
            f.write(f"Predicted GPA = {coef:.6f} * StudyHours + {intercept:.6f}\n")
        print("[saved] simple_linear_model_gpa.txt")
    return rows


def extended_linear(df, target_col: str, label: str):
    """Linear regression using multiple features to predict target_col."""
    feature_cols = ["StudyHours", "Absences", "Tutoring", "ParentalSupport",
                    "Extracurricular", "Sports", "Music", "Volunteering",
                    "Age", "Gender", "Ethnicity", "ParentalEducation"]
    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ext_mae, ext_mse, ext_rmse, ext_r2 = metrics(y_test, y_pred)

    return [{
        "Model": f"Linear (Extended Features)->{label}",
        "MAE": ext_mae, "MSE": ext_mse, "RMSE": ext_rmse, "R2": ext_r2
    }]


def main():
    data_path_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    csv_path = ensure_path(data_path_arg)

    # Load
    df = pd.read_csv(csv_path)
    print(f"[loaded] {csv_path}  shape={df.shape}")

    # Basic rename for clarity
    if "StudyTimeWeekly" in df.columns:
        df = df.rename(columns={"StudyTimeWeekly": "StudyHours"})

    # Remove duplicate StudentIDs (safety)
    if "StudentID" in df.columns:
        df = df.drop_duplicates(subset=["StudentID"])

    # Clean outliers for StudyHours using IQR
    Q1, Q3 = df["StudyHours"].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df_clean = df[(df["StudyHours"] >= lower) & (df["StudyHours"] <= upper)].copy()
    print(f"[cleaning] StudyHours IQR filter: kept {len(df_clean)}/{len(df)} rows")

    # EDA
    describe_and_plot(df_clean)

    # Modeling for GPA
    gpa_rows = simple_linear_and_poly(df_clean, target_col="GPA", label="GPA")
    gpa_rows += extended_linear(df_clean, target_col="GPA", label="GPA")
    results_gpa = pd.DataFrame(gpa_rows)
    results_gpa_rounded = results_gpa.copy()
    for col in ["MAE", "MSE", "RMSE", "R2"]:
        results_gpa_rounded[col] = results_gpa_rounded[col].round(4)
    print("\n=== Model Performance (GPA target) ===")
    print(results_gpa_rounded)
    results_gpa.to_csv(os.path.join(OUTPUT_DIR, "model_performance_gpa.csv"), index=False)
    print(f"[saved] {os.path.join(OUTPUT_DIR, 'model_performance_gpa.csv')}")

    # Modeling for GradeClass (treat as exam score)
    gc_rows = simple_linear_and_poly(df_clean, target_col="GradeClass", label="GradeClass (Exam Score)")
    gc_rows += extended_linear(df_clean, target_col="GradeClass", label="GradeClass (Exam Score)")
    results_gc = pd.DataFrame(gc_rows)
    results_gc_rounded = results_gc.copy()
    for col in ["MAE", "MSE", "RMSE", "R2"]:
        results_gc_rounded[col] = results_gc_rounded[col].round(4)
    print("\n=== Model Performance (GradeClass target) ===")
    print(results_gc_rounded)
    results_gc.to_csv(os.path.join(OUTPUT_DIR, "model_performance_gradeclass.csv"), index=False)
    print(f"[saved] {os.path.join(OUTPUT_DIR, 'model_performance_gradeclass.csv')}")

    print("\nDone. Plots and CSVs saved in:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
