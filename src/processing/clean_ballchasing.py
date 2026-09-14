import pandas as pd
from pathlib import Path


RAW_FOLDER = Path("data/raw/ballchasing")
OUTPUT_FOLDER = Path("data/processed/ballchasing")


def load_team_csv(file_path):
    df = pd.read_csv(file_path, sep=";")

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    return df


def load_player_csv(file_path, team_df):
    df = pd.read_csv(file_path, sep=";")

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    # The player CSV does not contain team names.
    # Recover them using the colour field from the team CSV.
    colour_to_team = dict(
        zip(team_df["color"], team_df["team name"])
    )

    df["team name"] = df["color"].map(colour_to_team)

    return df


def process_all_games():
    print("=" * 60)
    print("BALLCHASING BATCH CLEANING")
    print("=" * 60)

    team_files = sorted(RAW_FOLDER.glob("*_team.csv"))

    if not team_files:
        print("\nNo team CSV files found.")
        return

    print(f"\nFound {len(team_files)} team CSV files.")

    all_team_data = []
    all_player_data = []

    for team_file in team_files:

        # Find matching player file
        player_file = RAW_FOLDER / team_file.name.replace(
            "_team.csv",
            "_player.csv"
        )

        if not player_file.exists():
            print(f"\nWARNING: No player file found for {team_file.name}")
            continue

        # Extract game number from filename
        game_number = int(
            team_file.stem.split("_game")[1].split("_")[0]
        )

        print(f"\nProcessing Game {game_number}...")
        print(f"  Team:   {team_file.name}")
        print(f"  Player: {player_file.name}")

        # Load files
        team_df = load_team_csv(team_file)
        player_df = load_player_csv(player_file, team_df)

        # Add game number
        team_df["game_number"] = game_number
        player_df["game_number"] = game_number

        # Add source filename
        team_df["source_file"] = team_file.name
        player_df["source_file"] = player_file.name

        all_team_data.append(team_df)
        all_player_data.append(player_df)

        print(f"  Team rows:   {len(team_df)}")
        print(f"  Player rows: {len(player_df)}")

    # Combine all games
    combined_team = pd.concat(
        all_team_data,
        ignore_index=True
    )

    combined_player = pd.concat(
        all_player_data,
        ignore_index=True
    )

    # Create output folder
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Output files
    team_output = OUTPUT_FOLDER / "clean_team_all_games.csv"
    player_output = OUTPUT_FOLDER / "clean_player_all_games.csv"

    combined_team.to_csv(
        team_output,
        index=False
    )

    combined_player.to_csv(
        player_output,
        index=False
    )

    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)

    print(f"\nCombined team data:")
    print(f"  Rows: {len(combined_team)}")
    print(f"  Columns: {len(combined_team.columns)}")

    print(f"\nCombined player data:")
    print(f"  Rows: {len(combined_player)}")
    print(f"  Columns: {len(combined_player.columns)}")

    print("\nGames processed:")
    print(sorted(combined_team["game_number"].unique()))

    print("\nFiles saved:")
    print(f"  {team_output}")
    print(f"  {player_output}")


if __name__ == "__main__":
    process_all_games()