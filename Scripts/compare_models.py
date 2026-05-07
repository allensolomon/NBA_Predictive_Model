import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def main():
    path = "../data/processed/training_table.csv"
    df = pd.read_csv(path)

    target_col = "TARGET_3P_PCT"

    baseline_features = ["3P_PCT"]

    final_features = [
        "AGE",
        "3PA",
        "SHOT_FREQUENCY_PCT",
        "3P_PCT",
        "LEFT_CORNER_3_PCT",
        "RIGHT_CORNER_3_PCT",
        "ABOVE_BREAK_3_PCT",
        "CORNER_3_PCT"
    ]

    keep_cols = list(set(baseline_features + final_features + [target_col]))
    model_df = df[keep_cols].copy()

    for col in keep_cols:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

    model_df = model_df.replace([np.inf, -np.inf], np.nan)
    model_df = model_df.dropna(subset=[target_col]).copy()

    train_idx, test_idx = train_test_split(
        model_df.index, test_size=0.2, random_state=42
    )

    train_df = model_df.loc[train_idx].copy()
    test_df = model_df.loc[test_idx].copy()

    X_train_base = train_df[baseline_features]
    X_test_base = test_df[baseline_features]
    y_train = train_df[target_col]
    y_test = test_df[target_col]

    baseline_model = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("linear", LinearRegression())
    ])

    baseline_model.fit(X_train_base, y_train)
    baseline_preds = baseline_model.predict(X_test_base)
    baseline_rmse = rmse(y_test, baseline_preds)

    X_train_final = train_df[final_features]
    X_test_final = test_df[final_features]

    final_model = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("ridge", Ridge(alpha=1.0))
    ])

    final_model.fit(X_train_final, y_train)
    final_preds = final_model.predict(X_test_final)
    final_rmse = rmse(y_test, final_preds)

    print("Model Comparison")
    print("----------------")
    print(f"Baseline RMSE (3P_PCT only): {baseline_rmse:.4f}")
    print(f"Final RMSE (engineered features): {final_rmse:.4f}")
    print(f"Improvement in RMSE: {baseline_rmse - final_rmse:.4f}")

    ridge_model = final_model.named_steps["ridge"]
    print("\nFinal Model Coefficients:")
    for col, coef in zip(final_features, ridge_model.coef_):
        print(f"{col}: {coef:.4f}")


if __name__ == "__main__":
    main()
