import time
import pandas as pd
from nba_api.stats.endpoints import leaguedashplayershotlocations

SEASONS = [
    "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"
]


def get_zone_data_for_season(season: str) -> pd.DataFrame:
    print(f"Pulling zone data for {season}...")

    resp = leaguedashplayershotlocations.LeagueDashPlayerShotLocations(
        season=season,
        distance_range="By Zone"
    )

    data = resp.get_dict()
    result_sets = data["resultSets"]

    if isinstance(result_sets, list):
        rs = result_sets[0]
    else:
        rs = result_sets

    headers = rs["headers"]
    rows = rs["rowSet"]

    shot_categories = [
        "Restricted Area",
        "In The Paint (Non-RA)",
        "Mid-Range",
        "Left Corner 3",
        "Right Corner 3",
        "Above the Break 3",
        "Backcourt",
        "Corner 3"
    ]

    base_columns = headers[1]["columnNames"][:6]

    all_columns = base_columns.copy()

    for zone in shot_categories:
        safe_zone = (
            zone.upper()
            .replace(" ", "_")
            .replace("-", "")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
        )
        all_columns.extend([
            f"{safe_zone}_FGM",
            f"{safe_zone}_FGA",
            f"{safe_zone}_FG_PCT"
        ])

    df = pd.DataFrame(rows, columns=all_columns)
    df["SEASON"] = season

    keep_cols = [
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "AGE",
        "SEASON",
        "LEFT_CORNER_3_FGM",
        "LEFT_CORNER_3_FGA",
        "LEFT_CORNER_3_FG_PCT",
        "RIGHT_CORNER_3_FGM",
        "RIGHT_CORNER_3_FGA",
        "RIGHT_CORNER_3_FG_PCT",
        "ABOVE_THE_BREAK_3_FGM",
        "ABOVE_THE_BREAK_3_FGA",
        "ABOVE_THE_BREAK_3_FG_PCT",
        "CORNER_3_FGM",
        "CORNER_3_FGA",
        "CORNER_3_FG_PCT"
    ]

    df = df[keep_cols].copy()

    df = df.rename(columns={
        "PLAYER_NAME": "NAME",
        "TEAM_ABBREVIATION": "TEAM",
        "LEFT_CORNER_3_FG_PCT": "LEFT_CORNER_3_PCT",
        "RIGHT_CORNER_3_FG_PCT": "RIGHT_CORNER_3_PCT",
        "ABOVE_THE_BREAK_3_FG_PCT": "ABOVE_BREAK_3_PCT",
        "CORNER_3_FG_PCT": "CORNER_3_PCT"
    })

    return df


def main():
    all_dfs = []

    for season in SEASONS:
        df = get_zone_data_for_season(season)
        all_dfs.append(df)
        time.sleep(1)

    full_df = pd.concat(all_dfs, ignore_index=True)

    output_path = "../data/raw/zone_shooting_basic.csv"
    full_df.to_csv(output_path, index=False)

    print("\nSaved zone data.")
    print(f"Path: {output_path}")
    print(f"Shape: {full_df.shape}")
    print("\nColumns:")
    print(full_df.columns.tolist())
    print("\nPreview:")
    print(full_df.head())


if __name__ == "__main__":
    main()
