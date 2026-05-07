import time
import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats


# Seasons to include (NO 2025-26)
SEASONS = [
    "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"
]


def get_player_stats_for_season(season: str) -> pd.DataFrame:
    print(f"Pulling season {season}...")

    resp = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="Totals"
    )

    data = resp.get_dict()

    # Handle API structure safely
    result_set = data["resultSets"][0] if isinstance(
        data["resultSets"], list) else data["resultSets"]
    headers = result_set["headers"]
    rows = result_set["rowSet"]

    df = pd.DataFrame(rows, columns=headers)

    # Keep only needed columns
    keep_cols = [
        "PLAYER_ID",
        "PLAYER_NAME",
        "AGE",
        "TEAM_ABBREVIATION",
        "FGA",
        "FG3A",
        "FG3_PCT"
    ]

    df = df[keep_cols].copy()

    # Add season
    df["SEASON"] = season

    # Shot frequency = 3PA / total FGA
    df["SHOT_FREQUENCY_PCT"] = df["FG3A"] / df["FGA"]

    # Rename columns to match project spec
    df = df.rename(columns={
        "PLAYER_NAME": "NAME",
        "TEAM_ABBREVIATION": "TEAM",
        "FG3A": "3PA",
        "FG3_PCT": "3P_PCT"
    })

    return df


def main():
    all_dfs = []

    for season in SEASONS:
        df = get_player_stats_for_season(season)
        all_dfs.append(df)

        # Avoid rate limiting
        time.sleep(1)

    final_df = pd.concat(all_dfs, ignore_index=True)

    # Clean edge cases
    final_df = final_df.dropna(subset=["3P_PCT"])
    final_df = final_df[final_df["FGA"] > 0]

    # Save to your data/raw folder
    output_path = "../data/raw/core_player_season_stats.csv"
    final_df.to_csv(output_path, index=False)

    print("\n Dataset created successfully!")
    print(f"Saved to: {output_path}")
    print(f"Shape: {final_df.shape}")
    print("\nPreview:")
    print(final_df.head())


if __name__ == "__main__":
    main()
