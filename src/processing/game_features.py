import pandas as pd


TEAM_FILE = "data/processed/ballchasing/clean_team_all_games.csv"
PLAYER_FEATURE_FILE = (
    "data/processed/ballchasing/player_team_features_all_games.csv"
)
METADATA_FILE = "data/processed/metadata/replay_metadata.csv"

OUTPUT_FILE = "data/processed/ballchasing/game_features.csv"


def create_game_features():

    print("=" * 60)
    print("GAME-LEVEL FEATURE CREATION")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    team_df = pd.read_csv(TEAM_FILE)
    player_df = pd.read_csv(PLAYER_FEATURE_FILE)
    metadata_df = pd.read_csv(METADATA_FILE)

    print(f"\nTeam rows: {len(team_df)}")
    print(f"Player feature rows: {len(player_df)}")
    print(f"Metadata rows: {len(metadata_df)}")

    # ---------------------------------------------------------
    # Prepare team statistics
    # ---------------------------------------------------------

    team_features = team_df.copy()

    # We only need the statistics that will be useful at this stage.
    team_columns = [
        "game_number",
        "team name",
        "score",
        "goals",
        "assists",
        "saves",
        "shots",
        "shots conceded",
        "goals conceded",
        "shooting percentage",
        "bpm",
        "avg boost amount",
        "amount collected",
        "amount stolen",
        "total distance",
        "time boost speed",
        "time supersonic speed",
        "time defensive third",
        "time neutral third",
        "time offensive third",
        "time ball possession",
        "demos inflicted",
        "demos taken",
    ]

    team_features = team_features[team_columns]

    # ---------------------------------------------------------
    # Prepare player-derived team features
    # ---------------------------------------------------------

    player_features = player_df.copy()

    # ---------------------------------------------------------
    # Start with replay metadata
    # ---------------------------------------------------------

    games = metadata_df.copy()

    # ---------------------------------------------------------
    # Merge Team A statistics
    # ---------------------------------------------------------

    team_a = team_features.rename(
        columns={
            "team name": "team_a",
            "score": "team_a_score_stat",
            "goals": "team_a_goals",
            "assists": "team_a_assists",
            "saves": "team_a_saves",
            "shots": "team_a_shots",
            "shots conceded": "team_a_shots_conceded",
            "goals conceded": "team_a_goals_conceded",
            "shooting percentage": "team_a_shooting_percentage",
            "bpm": "team_a_bpm",
            "avg boost amount": "team_a_avg_boost",
            "amount collected": "team_a_boost_collected",
            "amount stolen": "team_a_boost_stolen",
            "total distance": "team_a_total_distance",
            "time boost speed": "team_a_time_boost_speed",
            "time supersonic speed": "team_a_time_supersonic_speed",
            "time defensive third": "team_a_time_defensive_third",
            "time neutral third": "team_a_time_neutral_third",
            "time offensive third": "team_a_time_offensive_third",
            "time ball possession": "team_a_time_ball_possession",
            "demos inflicted": "team_a_demos_inflicted",
            "demos taken": "team_a_demos_taken",
        }
    )

    games = games.merge(
        team_a,
        on=["game_number", "team_a"],
        how="left"
    )

    # ---------------------------------------------------------
    # Merge Team B statistics
    # ---------------------------------------------------------

    team_b = team_features.rename(
        columns={
            "team name": "team_b",
            "score": "team_b_score_stat",
            "goals": "team_b_goals",
            "assists": "team_b_assists",
            "saves": "team_b_saves",
            "shots": "team_b_shots",
            "shots conceded": "team_b_shots_conceded",
            "goals conceded": "team_b_goals_conceded",
            "shooting percentage": "team_b_shooting_percentage",
            "bpm": "team_b_bpm",
            "avg boost amount": "team_b_avg_boost",
            "amount collected": "team_b_boost_collected",
            "amount stolen": "team_b_boost_stolen",
            "total distance": "team_b_total_distance",
            "time boost speed": "team_b_time_boost_speed",
            "time supersonic speed": "team_b_time_supersonic_speed",
            "time defensive third": "team_b_time_defensive_third",
            "time neutral third": "team_b_time_neutral_third",
            "time offensive third": "team_b_time_offensive_third",
            "time ball possession": "team_b_time_ball_possession",
            "demos inflicted": "team_b_demos_inflicted",
            "demos taken": "team_b_demos_taken",
        }
    )

    games = games.merge(
        team_b,
        on=["game_number", "team_b"],
        how="left"
    )

    # ---------------------------------------------------------
    # Merge player-derived features for Team A
    # ---------------------------------------------------------

    player_a = player_features.rename(
        columns={
            "team name": "team_a",
            "avg_player_score": "team_a_avg_player_score",
            "avg_player_goals": "team_a_avg_player_goals",
            "avg_player_assists": "team_a_avg_player_assists",
            "avg_player_saves": "team_a_avg_player_saves",
            "avg_player_shots": "team_a_avg_player_shots",
            "avg_player_shooting_percentage":
                "team_a_avg_player_shooting_percentage",
            "avg_player_boost": "team_a_avg_player_boost",
            "avg_player_demos_inflicted":
                "team_a_avg_player_demos_inflicted",
            "avg_player_demos_taken":
                "team_a_avg_player_demos_taken",
        }
    )

    games = games.merge(
        player_a,
        on=["game_number", "team_a"],
        how="left"
    )

    # ---------------------------------------------------------
    # Merge player-derived features for Team B
    # ---------------------------------------------------------

    player_b = player_features.rename(
        columns={
            "team name": "team_b",
            "avg_player_score": "team_b_avg_player_score",
            "avg_player_goals": "team_b_avg_player_goals",
            "avg_player_assists": "team_b_avg_player_assists",
            "avg_player_saves": "team_b_avg_player_saves",
            "avg_player_shots": "team_b_avg_player_shots",
            "avg_player_shooting_percentage":
                "team_b_avg_player_shooting_percentage",
            "avg_player_boost": "team_b_avg_player_boost",
            "avg_player_demos_inflicted":
                "team_b_avg_player_demos_inflicted",
            "avg_player_demos_taken":
                "team_b_avg_player_demos_taken",
        }
    )

    games = games.merge(
        player_b,
        on=["game_number", "team_b"],
        how="left"
    )

    # ---------------------------------------------------------
    # Create game winner
    # ---------------------------------------------------------

    games["winner"] = games.apply(
        lambda row: (
            row["team_a"]
            if row["game_team_a_score"] > row["game_team_b_score"]
            else row["team_b"]
        ),
        axis=1
    )

    # Binary target:
    # 1 = Team A wins
    # 0 = Team B wins

    games["target"] = (
        games["winner"] == games["team_a"]
    ).astype(int)

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    print("\nGame-level dataset:")
    print(f"Rows: {len(games)}")
    print(f"Columns: {len(games.columns)}")

    print("\nGame results:")

    print(
        games[
            [
                "game_number",
                "team_a",
                "team_b",
                "game_team_a_score",
                "game_team_b_score",
                "winner",
                "target",
            ]
        ].to_string(index=False)
    )

    # Check that every game found team statistics
    team_a_missing = games["team_a_goals"].isna().sum()
    team_b_missing = games["team_b_goals"].isna().sum()

    print("\nTeam statistic validation:")
    print(f"Missing Team A statistics: {team_a_missing}")
    print(f"Missing Team B statistics: {team_b_missing}")

    # Check player-derived features
    player_a_missing = games["team_a_avg_player_score"].isna().sum()
    player_b_missing = games["team_b_avg_player_score"].isna().sum()

    print("\nPlayer feature validation:")
    print(f"Missing Team A player features: {player_a_missing}")
    print(f"Missing Team B player features: {player_b_missing}")

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    games.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSaved to:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 60)
    print("GAME FEATURE CREATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    create_game_features()