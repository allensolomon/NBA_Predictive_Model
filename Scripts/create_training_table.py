import pandas as pd


def season_to_start_year(season: str) -> int:
    return int(season.split("-")[0])


def start_year_to_season(year: int) -> str:
    return f"{year}-{str(year + 1)[-2:]}"


def main():
    input_path = "../data/processed/modeling_dataset.csv"
    output_path = "../data/processed/training_table.csv"

    df = pd.read_csv(input_path)

    # Sort for clean shifting
    df["SEASON_START"] = df["SEASON"].apply(season_to_start_year)
    df = df.sort_values(["PLAYER_ID", "SEASON_START"]).reset_index(drop=True)

    # Create next-season target within each player
    df["TARGET_3P_PCT"] = df.groupby("PLAYER_ID")["3P_PCT"].shift(-1)
    df["NEXT_SEASON_START"] = df.groupby("PLAYER_ID")["SEASON_START"].shift(-1)

    # Keep only rows where the next season is actually the immediate next year
    df["EXPECTED_NEXT_SEASON_START"] = df["SEASON_START"] + 1
    train_df = df[df["NEXT_SEASON_START"] ==
                  df["EXPECTED_NEXT_SEASON_START"]].copy()

    # Optional: keep a readable next season label
    train_df["TARGET_SEASON"] = train_df["EXPECTED_NEXT_SEASON_START"].apply(
        start_year_to_season)

    # Drop helper columns you do not want in final training file
    train_df = train_df.drop(columns=[
        "NEXT_SEASON_START",
        "EXPECTED_NEXT_SEASON_START"
    ])

    print("Original modeling dataset shape:", df.shape)
    print("Training table shape:", train_df.shape)
    print("\nColumns:")
    print(train_df.columns.tolist())
    print("\nPreview:")
    print(train_df.head())

    train_df.to_csv(output_path, index=False)
    print(f"\nSaved training table to: {output_path}")


if __name__ == "__main__":
    main()
