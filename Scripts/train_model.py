import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


def main():
    path = "../data/processed/training_table.csv"
    df = pd.read_csv(path)

    feature_cols = [
        "AGE",
        "3PA",
        "SHOT_FREQUENCY_PCT",
        "3P_PCT",
        "LEFT_CORNER_3_PCT",
        "RIGHT_CORNER_3_PCT",
        "ABOVE_BREAK_3_PCT",
        "CORNER_3_PCT"
    ]
    target_col = "TARGET_3P_PCT"

    model_df = df[feature_cols + [target_col]].copy()

    for col in feature_cols + [target_col]:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

    model_df = model_df.replace([np.inf, -np.inf], np.nan)
    model_df = model_df.dropna(subset=[target_col]).copy()

    X = model_df[feature_cols]
    y = model_df[target_col]

    print("NaNs in X before split:")
    print(X.isna().sum())
    print("\nNaNs in y:")
    print(y.isna().sum())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nNaNs in X_train before model:")
    print(X_train.isna().sum())

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("ridge", Ridge(alpha=1.0))
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"\nRMSE: {rmse:.4f}")

    ridge_model = model.named_steps["ridge"]
    print("\nFeature Coefficients:")
    for col, coef in zip(feature_cols, ridge_model.coef_):
        print(f"{col}: {coef:.4f}")


if __name__ == "__main__":
    main()
