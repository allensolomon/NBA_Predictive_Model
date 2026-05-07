import pandas as pd


def main():
    core_path = "../data/raw/core_player_season_stats.csv"
    zone_path = "../data/raw/zone_shooting_basic.csv"
    output_path = "../data/processed/modeling_dataset.csv"

    core_df = pd.read_csv(core_path)
    zone_df = pd.read_csv(zone_path)

    print("Core shape:", core_df.shape)
    print("Zone shape:", zone_df.shape)

    merged_df = core_df.merge(
        zone_df,
        on=["PLAYER_ID", "SEASON"],
        how="left",
        suffixes=("", "_zone")
    )

    # Keep the cleaner team/name fields from core_df
    drop_cols = [col for col in merged_df.columns if col.endswith("_zone")]
    merged_df = merged_df.drop(columns=drop_cols, errors="ignore")

    print("Merged shape:", merged_df.shape)
    print("\nColumns:")
    print(merged_df.columns.tolist())
    print("\nPreview:")
    print(merged_df.head())

    merged_df.to_csv(output_path, index=False)
    print(f"\nSaved merged dataset to: {output_path}")


if __name__ == "__main__":
    main()
