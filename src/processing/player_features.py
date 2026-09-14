import pandas as pd
from pathlib import Path


INPUT_FILE = "data/processed/ballchasing/clean_player_all_games.csv"
OUTPUT_FILE = "data/processed/ballchasing/player_team_features_all_games.csv"


def create_team_player_features(df):

    features = (
        df.groupby(["game_number", "team name"])
        .agg(
            avg_player_score=("score", "mean"),
            avg_player_goals=("goals", "mean"),
            avg_player_assists=("assists", "mean"),
            avg_player_saves=("saves", "mean"),
            avg_player_shots=("shots", "mean"),
            avg_player_shooting_percentage=("shooting percentage", "mean"),
            avg_player_boost=("avg boost amount", "mean"),
            avg_player_demos_inflicted=("demos inflicted", "mean"),
            avg_player_demos_taken=("demos taken", "mean"),
        )
        .reset_index()
    )

    return features


if __name__ == "__main__":

    print("=" * 60)
    print("PLAYER → TEAM FEATURE AGGREGATION")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)

    print(f"\nPlayer rows: {len(df)}")
    print(f"Games: {sorted(df['game_number'].unique())}")

    features = create_team_player_features(df)

    print("\nTeam-level player features:")
    print(features.to_string(index=False))

    features.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSaved to:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 60)
    print("FEATURE CREATION COMPLETE")
    print("=" * 60)