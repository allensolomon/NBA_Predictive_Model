import time
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import shotchartdetail


def get_player_shot_profile(player_id, season):
    resp = shotchartdetail.ShotChartDetail(
        team_id=0,
        player_id=player_id,
        season_nullable=season,
        context_measure_simple="FGA"
    )

    data = resp.get_dict()
    rs = data["resultSets"][0]

    headers = rs["headers"]
    rows = rs["rowSet"]

    df = pd.DataFrame(rows, columns=headers)

    df_3 = df[df["SHOT_TYPE"] == "3PT Field Goal"].copy()

    left_wing = df_3[
        (df_3["SHOT_ZONE_BASIC"] == "Above the Break 3") &
        (df_3["SHOT_ZONE_AREA"].isin(["Left Side(L)", "Left Side Center(LC)"]))
    ]

    right_wing = df_3[
        (df_3["SHOT_ZONE_BASIC"] == "Above the Break 3") &
        (df_3["SHOT_ZONE_AREA"].isin(["Right Side(R)", "Right Side Center(RC)"]))
    ]

    corners = df_3[
        df_3["SHOT_ZONE_BASIC"].isin(["Left Corner 3", "Right Corner 3"])
    ]

    def calc_pct(d):
        if len(d) == 0:
            return np.nan
        return float(d["SHOT_MADE_FLAG"].mean())

    return {
        "PLAYER_ID": player_id,
        "SEASON": season,
        "LEFT_WING_3_PCT": calc_pct(left_wing),
        "RIGHT_WING_3_PCT": calc_pct(right_wing),
        "CORNER_3_PCT_DETAIL": calc_pct(corners)
    }


def get_with_retries(player_id, season, max_attempts=3):
    for attempt in range(1, max_attempts + 1):
        try:
            return get_player_shot_profile(player_id, season)
        except Exception as e:
            print(
                f"Retry {attempt}/{max_attempts} failed "
                f"for player_id={player_id}, season={season}: {e}"
            )
            time.sleep(2 * attempt)

    raise RuntimeError(
        f"All retries failed for player_id={player_id}, season={season}")


def main():
    core_path = "../data/raw/core_player_season_stats.csv"
    output_path = "../data/raw/shot_profile_wings.csv"
    checkpoint_path = "../data/raw/shot_profile_wings_checkpoint.csv"

    core_df = pd.read_csv(core_path)

    # Optional but smart: skip tiny-volume shooters for this enhancement step
    core_df = core_df[core_df["3PA"] >= 50].copy()

    player_seasons = (
        core_df[["PLAYER_ID", "SEASON"]]
        .drop_duplicates()
        .sort_values(["SEASON", "PLAYER_ID"])
        .reset_index(drop=True)
    )

    results = []

    for i, row in player_seasons.iterrows():
        player_id = int(row["PLAYER_ID"])
        season = row["SEASON"]

        try:
            profile = get_with_retries(player_id, season, max_attempts=3)
            results.append(profile)
            print(
                f"[{i+1}/{len(player_seasons)}] OK   player_id={player_id}, season={season}")
        except Exception as e:
            print(
                f"[{i+1}/{len(player_seasons)}] FAIL player_id={player_id}, season={season}, error={e}")
            results.append({
                "PLAYER_ID": player_id,
                "SEASON": season,
                "LEFT_WING_3_PCT": np.nan,
                "RIGHT_WING_3_PCT": np.nan,
                "CORNER_3_PCT_DETAIL": np.nan
            })

        # Save checkpoint every 25 rows
        if (i + 1) % 25 == 0:
            pd.DataFrame(results).to_csv(checkpoint_path, index=False)
            print(f"Checkpoint saved: {checkpoint_path}")

        # Slower pacing to reduce failures
        time.sleep(1.5)

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)

    print(f"\nSaved shot profile dataset to: {output_path}")
    print("Shape:", out_df.shape)
    print("\nPreview:")
    print(out_df.head())


if __name__ == "__main__":
    main()
