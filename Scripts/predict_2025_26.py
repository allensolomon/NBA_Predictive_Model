import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def main():
    train_path = "../data/processed/training_table.csv"
    modeling_path = "../data/processed/modeling_dataset.csv"
    output_path = "../data/processed/predictions_2025_26.csv"

    train_df = pd.read_csv(train_path)
    modeling_df = pd.read_csv(modeling_path)

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

    # --- TRAIN MODEL ---
    model_df = train_df[feature_cols + [target_col]].copy()

    for col in feature_cols + [target_col]:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

    model_df = model_df.replace([np.inf, -np.inf], np.nan)
    model_df = model_df.dropna(subset=[target_col]).copy()

    X_train = model_df[feature_cols]
    y_train = model_df[target_col]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("ridge", Ridge(alpha=1.0))
    ])

    model.fit(X_train, y_train)

    # PREDICT FOR 2025-26 (using 2024-25 data)
    pred_df = modeling_df[modeling_df["SEASON"] == "2024-25"].copy()

    for col in feature_cols:
        pred_df[col] = pd.to_numeric(pred_df[col], errors="coerce")

    X_pred = pred_df[feature_cols]

    pred_df["PREDICTED_3P_PCT_2025_26"] = model.predict(X_pred)

    # Clip to valid range
    pred_df["PREDICTED_3P_PCT_2025_26"] = pred_df["PREDICTED_3P_PCT_2025_26"].clip(
        0, 1)

    # Filter low-volume shooters
    pred_df = pred_df[pred_df["3PA"] >= 100]

    # Round predictions
    pred_df["PREDICTED_3P_PCT_2025_26"] = pred_df["PREDICTED_3P_PCT_2025_26"].round(
        3)

    submission_cols = [
        "PLAYER_ID",
        "NAME",
        "TEAM",
        "AGE",
        "3PA",
        "3P_PCT",
        "PREDICTED_3P_PCT_2025_26"
    ]

    output_df = pred_df[submission_cols].copy()
    output_df = output_df.sort_values(
        "PREDICTED_3P_PCT_2025_26", ascending=False
    )

    output_df.to_csv(output_path, index=False)

    print(f"Saved predictions to: {output_path}")
    print(f"Shape: {output_df.shape}")
    print("\nTop 10 predictions:")
    print(output_df.head(10))


if __name__ == "__main__":
    main()
