import pandas as pd


INPUT_FILE = r"data\processed\ballchasing\open1_games_with_time.csv"
OUTPUT_FILE = r"data\processed\ballchasing\open1_series_candidates.csv"


def main():

    print("Loading Open 1 games...")
    df = pd.read_csv(INPUT_FILE)

    df["replay_datetime"] = pd.to_datetime(
        df["replay_datetime"],
        errors="coerce"
    )

    # Create a consistent team-pair identifier.
    df["team_1"] = df[["blue_team", "orange_team"]].min(axis=1)
    df["team_2"] = df[["blue_team", "orange_team"]].max(axis=1)

    df["team_pair"] = (
        df["team_1"].astype(str)
        + " vs "
        + df["team_2"].astype(str)
    )

    # Sort by stage/subgroup/time.
    df = df.sort_values(
        [
            "stage",
            "subgroup",
            "replay_datetime"
        ]
    ).reset_index(drop=True)

    # A new candidate series starts when:
    # - the team pair changes, OR
    # - there is a long time gap between games.
    #
    # We use 30 minutes as a conservative initial threshold.
    # This creates candidates for inspection rather than claiming
    # these are definitively the tournament series.
    df["previous_team_pair"] = (
        df.groupby(
            ["stage", "subgroup"]
        )["team_pair"]
        .shift(1)
    )

    df["previous_time"] = (
        df.groupby(
            ["stage", "subgroup"]
        )["replay_datetime"]
        .shift(1)
    )

    df["minutes_since_previous"] = (
        (
            df["replay_datetime"]
            - df["previous_time"]
        ).dt.total_seconds() / 60
    )

    df["new_candidate_series"] = (
        df["team_pair"] != df["previous_team_pair"]
    ) | (
        df["minutes_since_previous"] > 30
    )

    df["candidate_series_number"] = (
        df.groupby(
            ["stage", "subgroup"]
        )["new_candidate_series"]
        .cumsum()
    )

    df["candidate_series_id"] = (
        df["stage"].astype(str)
        + "_"
        + df["subgroup"].astype(str)
        + "_S"
        + df["candidate_series_number"].astype(str)
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("Saved:")
    print(OUTPUT_FILE)

    print()
    print(
        "Candidate series:",
        df["candidate_series_id"].nunique()
    )

    print()
    print("Games per candidate series:")
    print(
        df.groupby(
            "candidate_series_id"
        ).size().value_counts().sort_index()
    )

    print()
    print("Candidate series summary:")
    
    summary = (
        df.groupby(
            [
                "candidate_series_id",
                "stage",
                "subgroup",
                "team_pair"
            ]
        )
        .agg(
            games=("replay_id", "count"),
            first_game=("replay_datetime", "min"),
            last_game=("replay_datetime", "max")
        )
        .reset_index()
    )

    print(
        summary.to_string(index=False)
    )


if __name__ == "__main__":
    main()