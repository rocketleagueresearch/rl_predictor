from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ballchasing" / "Open 1"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "ballchasing"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_column_names(df):
    """
    Strip whitespace from column names.
    """
    df.columns = [
        str(col).strip().lower()
        for col in df.columns
    ]

    return df


def extract_replay_id(filename):
    """
    Extract the Ballchasing replay UUID from filenames such as:

        015c3705-4035-4488-9cbe-7d6102530e43-teams.csv

    Also works with the older test filenames, although those do
    not contain a UUID.
    """

    suffix = "-teams.csv"

    if filename.lower().endswith(suffix):
        return filename[:-len(suffix)]

    suffix = "_team.csv"

    if filename.lower().endswith(suffix):
        return filename[:-len(suffix)]

    return filename


def classify_folder(team_file):
    """
    Determine tournament stage/group from the folder structure.

    Expected examples:

        Group Stages / Group A
        Group Stages / Group B
        Playoff 1
        Playoff 2
    """

    relative = team_file.relative_to(RAW_DIR)
    parts = relative.parts

    stage_folder = parts[0] if len(parts) >= 1 else ""

    if stage_folder == "Group Stages":

        group = parts[1] if len(parts) >= 3 else ""

        return stage_folder, group

    return stage_folder, ""


def safe_numeric(value):
    """
    Convert a value to numeric where possible.
    """
    return pd.to_numeric(value, errors="coerce")


# ============================================================
# MAIN PROCESSING
# ============================================================

def main():

    print("=" * 70)
    print("PROCESSING BALLCHASING OPEN 1 GAMES")
    print("=" * 70)

    if not RAW_DIR.exists():
        print("\nERROR: Open 1 directory not found:")
        print(RAW_DIR)
        return

    # --------------------------------------------------------
    # Find team CSV files
    # --------------------------------------------------------

    team_files = sorted(
        [
            f
            for f in RAW_DIR.rglob("*.csv")
            if (
                f.name.lower().endswith("-teams.csv")
                or
                f.name.lower().endswith("_team.csv")
            )
        ]
    )

    print(f"\nTeam CSV files found: {len(team_files)}")

    rows = []

    # --------------------------------------------------------
    # Process each game
    # --------------------------------------------------------

    for number, team_file in enumerate(team_files, start=1):

        try:

            # Ballchasing exports use semicolon separators
            df = pd.read_csv(
                team_file,
                sep=";"
            )

            df = clean_column_names(df)

            # ------------------------------------------------
            # Basic validation
            # ------------------------------------------------

            required_columns = [
                "color",
                "team name",
                "game duration",
                "score",
                "goals",
                "assists",
                "saves",
                "shots",
                "goals conceded",
                "shooting percentage"
            ]

            missing_columns = [
                col
                for col in required_columns
                if col not in df.columns
            ]

            if missing_columns:

                print(
                    f"\nWARNING: Missing columns in "
                    f"{team_file.name}:"
                )

                print(missing_columns)

                continue

            if len(df) != 2:

                print(
                    f"\nWARNING: Expected 2 team rows but found "
                    f"{len(df)} in {team_file.name}"
                )

                continue

            # ------------------------------------------------
            # Sort team rows consistently
            #
            # We use the CSV's blue/orange designation only
            # for identifying the two sides. Team A/B will be
            # normalised later when we build the series table.
            # ------------------------------------------------

            blue = df[
                df["color"].astype(str).str.lower() == "blue"
            ]

            orange = df[
                df["color"].astype(str).str.lower() == "orange"
            ]

            if len(blue) != 1 or len(orange) != 1:

                print(
                    f"\nWARNING: Could not identify blue/orange "
                    f"teams in {team_file.name}"
                )

                continue

            blue = blue.iloc[0]
            orange = orange.iloc[0]

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            replay_id = extract_replay_id(
                team_file.name
            )

            stage, subgroup = classify_folder(
                team_file
            )

            # ------------------------------------------------
            # Create one row per game
            # ------------------------------------------------

            row = {

                # Identification
                "replay_id": replay_id,

                # Tournament information
                "open": "Open 1",
                "stage": stage,
                "subgroup": subgroup,

                # Source information
                "source_team_file": str(
                    team_file.relative_to(PROJECT_ROOT)
                ),

                # Team identities
                "blue_team": blue["team name"],
                "orange_team": orange["team name"],

                # Scores
                "blue_score": safe_numeric(
                    blue["score"]
                ),

                "orange_score": safe_numeric(
                    orange["score"]
                ),

                # Game duration
                "game_duration": safe_numeric(
                    blue["game duration"]
                ),

                # ------------------------------------------------
                # Blue team statistics
                # ------------------------------------------------

                "blue_goals": safe_numeric(
                    blue["goals"]
                ),

                "blue_assists": safe_numeric(
                    blue["assists"]
                ),

                "blue_saves": safe_numeric(
                    blue["saves"]
                ),

                "blue_shots": safe_numeric(
                    blue["shots"]
                ),

                "blue_shooting_percentage": safe_numeric(
                    blue["shooting percentage"]
                ),

                "blue_bpm": safe_numeric(
                    blue["bpm"]
                ),

                "blue_avg_boost": safe_numeric(
                    blue["avg boost amount"]
                ),

                "blue_amount_collected": safe_numeric(
                    blue["amount collected"]
                ),

                "blue_amount_stolen": safe_numeric(
                    blue["amount stolen"]
                ),

                "blue_total_distance": safe_numeric(
                    blue["total distance"]
                ),

                "blue_time_ball_possession": safe_numeric(
                    blue["time ball possession"]
                ),

                "blue_time_in_front_ball": safe_numeric(
                    blue["time in front of ball"]
                ),

                "blue_time_behind_ball": safe_numeric(
                    blue["time behind ball"]
                ),

                "blue_time_offensive_half": safe_numeric(
                    blue["time offensive half"]
                ),

                "blue_time_defensive_half": safe_numeric(
                    blue["time defensive half"]
                ),

                "blue_demos_inflicted": safe_numeric(
                    blue["demos inflicted"]
                ),

                "blue_demos_taken": safe_numeric(
                    blue["demos taken"]
                ),

                # ------------------------------------------------
                # Orange team statistics
                # ------------------------------------------------

                "orange_goals": safe_numeric(
                    orange["goals"]
                ),

                "orange_assists": safe_numeric(
                    orange["assists"]
                ),

                "orange_saves": safe_numeric(
                    orange["saves"]
                ),

                "orange_shots": safe_numeric(
                    orange["shots"]
                ),

                "orange_shooting_percentage": safe_numeric(
                    orange["shooting percentage"]
                ),

                "orange_bpm": safe_numeric(
                    orange["bpm"]
                ),

                "orange_avg_boost": safe_numeric(
                    orange["avg boost amount"]
                ),

                "orange_amount_collected": safe_numeric(
                    orange["amount collected"]
                ),

                "orange_amount_stolen": safe_numeric(
                    orange["amount stolen"]
                ),

                "orange_total_distance": safe_numeric(
                    orange["total distance"]
                ),

                "orange_time_ball_possession": safe_numeric(
                    orange["time ball possession"]
                ),

                "orange_time_in_front_ball": safe_numeric(
                    orange["time in front of ball"]
                ),

                "orange_time_behind_ball": safe_numeric(
                    orange["time behind ball"]
                ),

                "orange_time_offensive_half": safe_numeric(
                    orange["time offensive half"]
                ),

                "orange_time_defensive_half": safe_numeric(
                    orange["time defensive half"]
                ),

                "orange_demos_inflicted": safe_numeric(
                    orange["demos inflicted"]
                ),

                "orange_demos_taken": safe_numeric(
                    orange["demos taken"]
                ),
            }

            # ------------------------------------------------
            # Determine game winner
            # ------------------------------------------------

            if row["blue_score"] > row["orange_score"]:

                row["game_winner"] = row["blue_team"]

            elif row["orange_score"] > row["blue_score"]:

                row["game_winner"] = row["orange_team"]

            else:

                row["game_winner"] = ""

            rows.append(row)

        except Exception as e:

            print(
                f"\nERROR processing {team_file.name}:"
            )

            print(e)

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    games = pd.DataFrame(rows)

    if games.empty:

        print("\nERROR: No games were successfully processed.")

        return

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    games = games.sort_values(
        [
            "stage",
            "subgroup",
            "replay_id"
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = (
        OUTPUT_DIR /
        "open1_games.csv"
    )

    games.to_csv(
        output_file,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "-" * 70)
    print("PROCESSING RESULTS")
    print("-" * 70)

    print(
        f"Games successfully processed: {len(games)}"
    )

    print(
        f"Columns created: {len(games.columns)}"
    )

    print("\nGames by stage/group:")

    counts = (
        games
        .groupby(
            ["stage", "subgroup"],
            dropna=False
        )
        .size()
        .reset_index(
            name="games"
        )
    )

    print(
        counts.to_string(index=False)
    )

    print("\nGame winners:")

    winner_counts = (
        games["game_winner"]
        .value_counts()
        .head(10)
    )

    print(
        winner_counts.to_string()
    )

    print("\nOutput saved to:")

    print(output_file)

    print("\n" + "=" * 70)
    print("OPEN 1 GAME PROCESSING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()