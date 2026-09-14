from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BALLCHASING_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ballchasing"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ballchasing"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_OPEN2 = OUTPUT_DIR / "open2_games.csv"
OUTPUT_OPEN3 = OUTPUT_DIR / "open3_games.csv"


# ============================================================
# COLUMN HELPERS
# ============================================================

def first_existing_column(df, candidates):
    """
    Return the first matching column name, case-insensitively.
    """
    lower_map = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in lower_map:
            return lower_map[key]

    return None


def numeric_value(row, candidates):
    """
    Read a numeric value from the first matching column.
    """
    for candidate in candidates:
        if candidate in row.index:
            return pd.to_numeric(
                row[candidate],
                errors="coerce",
            )

    return np.nan


def safe_sum(series):
    """
    Sum a numeric series while preserving NaN when every value is missing.
    """
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    if values.notna().sum() == 0:
        return np.nan

    return values.sum()


def safe_mean(series):
    """
    Mean of a numeric series while preserving NaN when all values are missing.
    """
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    if values.notna().sum() == 0:
        return np.nan

    return values.mean()


# ============================================================
# FILE HELPERS
# ============================================================

def replay_id_from_team_file(path):
    name = path.name

    if name.lower().endswith("-teams.csv"):
        return name[:-10]

    if name.lower().endswith("_team.csv"):
        return name[:-9]

    return None


def matching_player_file(team_file):
    """
    Find the corresponding player file in the same folder.
    """
    replay_id = replay_id_from_team_file(team_file)

    if replay_id is None:
        return None

    candidates = [
        team_file.parent / f"{replay_id}-players.csv",
        team_file.parent / f"{replay_id}_player.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def stage_from_path(file_path, open_root):
    """
    Determine stage and subgroup from folder path.
    """

    relative_parts = file_path.relative_to(open_root).parts

    stage = ""
    subgroup = ""

    if len(relative_parts) >= 2:
        stage = relative_parts[0]

    if stage == "Group Stages" and len(relative_parts) >= 3:
        subgroup = relative_parts[1]

    return stage, subgroup


# ============================================================
# RAW CSV LOADING
# ============================================================

def read_ballchasing_csv(path):
    """
    Ballchasing exports are semicolon-delimited.
    """
    return pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
    )


# ============================================================
# TEAM NAME EXTRACTION
# ============================================================

def clean_team_name(value):
    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    if text == "":
        return np.nan

    return text


def extract_team_names(team_df, player_df):
    """
    Get blue/orange team names.

    Prefer team CSV team-name values when available.
    Fall back to player CSV team-name values.
    Do not infer team names from player identities.
    """

    blue_team = np.nan
    orange_team = np.nan

    color_col_team = first_existing_column(
        team_df,
        ["color", "team color"],
    )

    team_name_col_team = first_existing_column(
        team_df,
        ["team name", "name", "team"],
    )

    if (
        color_col_team is not None
        and team_name_col_team is not None
    ):
        colors = (
            team_df[color_col_team]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        blue_rows = team_df[colors == "blue"]
        orange_rows = team_df[colors == "orange"]

        if len(blue_rows) > 0:
            blue_team = clean_team_name(
                blue_rows.iloc[0][team_name_col_team]
            )

        if len(orange_rows) > 0:
            orange_team = clean_team_name(
                orange_rows.iloc[0][team_name_col_team]
            )

    color_col_player = first_existing_column(
        player_df,
        ["color", "team color"],
    )

    team_name_col_player = first_existing_column(
        player_df,
        ["team name", "team"],
    )

    if (
        color_col_player is not None
        and team_name_col_player is not None
    ):
        colors = (
            player_df[color_col_player]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        if pd.isna(blue_team):
            blue_values = (
                player_df.loc[
                    colors == "blue",
                    team_name_col_player,
                ]
                .dropna()
                .astype(str)
                .str.strip()
            )

            blue_values = blue_values[
                blue_values != ""
            ]

            if len(blue_values) > 0:
                blue_team = blue_values.iloc[0]

        if pd.isna(orange_team):
            orange_values = (
                player_df.loc[
                    colors == "orange",
                    team_name_col_player,
                ]
                .dropna()
                .astype(str)
                .str.strip()
            )

            orange_values = orange_values[
                orange_values != ""
            ]

            if len(orange_values) > 0:
                orange_team = orange_values.iloc[0]

    return blue_team, orange_team


# ============================================================
# TEAM ROW EXTRACTION
# ============================================================

def extract_team_row(team_df, color):
    color_col = first_existing_column(
        team_df,
        ["color", "team color"],
    )

    if color_col is None:
        return None

    mask = (
        team_df[color_col]
        .astype(str)
        .str.strip()
        .str.lower()
        == color.lower()
    )

    rows = team_df[mask]

    if len(rows) == 0:
        return None

    return rows.iloc[0]


# ============================================================
# PLAYER AGGREGATION
# ============================================================

def aggregate_player_side(player_df, color):
    """
    Aggregate player stats for one team side.
    Sums counting stats; averages rate/boost-style fields.
    """

    color_col = first_existing_column(
        player_df,
        ["color", "team color"],
    )

    if color_col is None:
        return {}

    mask = (
        player_df[color_col]
        .astype(str)
        .str.strip()
        .str.lower()
        == color.lower()
    )

    side = player_df[mask].copy()

    if len(side) == 0:
        return {}

    column_aliases = {
        "goals": ["goals"],
        "assists": ["assists"],
        "saves": ["saves"],
        "shots": ["shots"],
        "shooting_percentage": [
            "shooting %",
            "shooting percentage",
            "shooting_percentage",
        ],
        "bpm": ["bpm"],
        "avg_boost": [
            "avg boost",
            "average boost",
            "avg_boost",
        ],
        "amount_collected": [
            "amount collected",
            "boost amount collected",
            "amount_collected",
        ],
        "amount_stolen": [
            "amount stolen",
            "boost amount stolen",
            "amount_stolen",
        ],
        "total_distance": [
            "total distance",
            "distance",
            "total_distance",
        ],
        "time_ball_possession": [
            "time ball possession",
            "ball possession",
            "time_ball_possession",
        ],
        "time_in_front_ball": [
            "time in front ball",
            "time in front of ball",
            "time_in_front_ball",
        ],
        "time_behind_ball": [
            "time behind ball",
            "time behind of ball",
            "time_behind_ball",
        ],
        "time_offensive_half": [
            "time offensive half",
            "time in offensive half",
            "time_offensive_half",
        ],
        "time_defensive_half": [
            "time defensive half",
            "time in defensive half",
            "time_defensive_half",
        ],
        "demos_inflicted": [
            "demos inflicted",
            "demos inflicted on opponents",
            "demos_inflicted",
        ],
        "demos_taken": [
            "demos taken",
            "demos received",
            "demos_taken",
        ],
    }

    output = {}

    sum_fields = {
        "goals",
        "assists",
        "saves",
        "shots",
        "amount_collected",
        "amount_stolen",
        "total_distance",
        "time_ball_possession",
        "time_in_front_ball",
        "time_behind_ball",
        "time_offensive_half",
        "time_defensive_half",
        "demos_inflicted",
        "demos_taken",
    }

    mean_fields = {
        "shooting_percentage",
        "bpm",
        "avg_boost",
    }

    for output_name, candidates in column_aliases.items():

        source_col = first_existing_column(
            side,
            candidates,
        )

        if source_col is None:
            output[output_name] = np.nan
            continue

        if output_name in sum_fields:
            output[output_name] = safe_sum(
                side[source_col]
            )

        elif output_name in mean_fields:
            output[output_name] = safe_mean(
                side[source_col]
            )

        else:
            output[output_name] = np.nan

    return output


# ============================================================
# SCORE + DURATION EXTRACTION
# ============================================================

def extract_score(team_row):
    if team_row is None:
        return np.nan

    candidates = [
        "score",
        "goals",
    ]

    for candidate in candidates:
        for col in team_row.index:
            if str(col).strip().lower() == candidate:
                value = pd.to_numeric(
                    team_row[col],
                    errors="coerce",
                )

                if pd.notna(value):
                    return value

    return np.nan


def extract_game_duration(team_df, player_df):
    candidates = [
        "game duration",
        "duration",
    ]

    for df in [team_df, player_df]:
        col = first_existing_column(
            df,
            candidates,
        )

        if col is not None:
            values = (
                df[col]
                .dropna()
            )

            if len(values) > 0:
                return values.iloc[0]

    return np.nan


# ============================================================
# GAME PROCESSING
# ============================================================

def process_game(team_file, open_name, open_root):

    replay_id = replay_id_from_team_file(
        team_file
    )

    player_file = matching_player_file(
        team_file
    )

    if player_file is None:
        raise FileNotFoundError(
            f"No matching player file for {team_file}"
        )

    team_df = read_ballchasing_csv(
        team_file
    )

    player_df = read_ballchasing_csv(
        player_file
    )

    stage, subgroup = stage_from_path(
        team_file,
        open_root,
    )

    blue_team, orange_team = extract_team_names(
        team_df,
        player_df,
    )

    blue_team_row = extract_team_row(
        team_df,
        "blue",
    )

    orange_team_row = extract_team_row(
        team_df,
        "orange",
    )

    blue_score = extract_score(
        blue_team_row
    )

    orange_score = extract_score(
        orange_team_row
    )

    blue_stats = aggregate_player_side(
        player_df,
        "blue",
    )

    orange_stats = aggregate_player_side(
        player_df,
        "orange",
    )

    # Prefer goal totals from player stats for winner logic.
    blue_goals = blue_stats.get(
        "goals",
        np.nan,
    )

    orange_goals = orange_stats.get(
        "goals",
        np.nan,
    )

    if pd.notna(blue_goals) and pd.notna(orange_goals):

        if blue_goals > orange_goals:
            game_winner = blue_team

        elif orange_goals > blue_goals:
            game_winner = orange_team

        else:
            game_winner = np.nan

    else:
        game_winner = np.nan

    record = {
        "replay_id": replay_id,
        "open": open_name,
        "stage": stage,
        "subgroup": subgroup,
        "source_team_file": str(
            team_file.relative_to(PROJECT_ROOT)
        ),
        "blue_team": blue_team,
        "orange_team": orange_team,
        "blue_score": blue_score,
        "orange_score": orange_score,
        "game_duration": extract_game_duration(
            team_df,
            player_df,
        ),
        "blue_goals": blue_stats.get("goals", np.nan),
        "blue_assists": blue_stats.get("assists", np.nan),
        "blue_saves": blue_stats.get("saves", np.nan),
        "blue_shots": blue_stats.get("shots", np.nan),
        "blue_shooting_percentage": blue_stats.get(
            "shooting_percentage",
            np.nan,
        ),
        "blue_bpm": blue_stats.get("bpm", np.nan),
        "blue_avg_boost": blue_stats.get(
            "avg_boost",
            np.nan,
        ),
        "blue_amount_collected": blue_stats.get(
            "amount_collected",
            np.nan,
        ),
        "blue_amount_stolen": blue_stats.get(
            "amount_stolen",
            np.nan,
        ),
        "blue_total_distance": blue_stats.get(
            "total_distance",
            np.nan,
        ),
        "blue_time_ball_possession": blue_stats.get(
            "time_ball_possession",
            np.nan,
        ),
        "blue_time_in_front_ball": blue_stats.get(
            "time_in_front_ball",
            np.nan,
        ),
        "blue_time_behind_ball": blue_stats.get(
            "time_behind_ball",
            np.nan,
        ),
        "blue_time_offensive_half": blue_stats.get(
            "time_offensive_half",
            np.nan,
        ),
        "blue_time_defensive_half": blue_stats.get(
            "time_defensive_half",
            np.nan,
        ),
        "blue_demos_inflicted": blue_stats.get(
            "demos_inflicted",
            np.nan,
        ),
        "blue_demos_taken": blue_stats.get(
            "demos_taken",
            np.nan,
        ),
        "orange_goals": orange_stats.get("goals", np.nan),
        "orange_assists": orange_stats.get("assists", np.nan),
        "orange_saves": orange_stats.get("saves", np.nan),
        "orange_shots": orange_stats.get("shots", np.nan),
        "orange_shooting_percentage": orange_stats.get(
            "shooting_percentage",
            np.nan,
        ),
        "orange_bpm": orange_stats.get("bpm", np.nan),
        "orange_avg_boost": orange_stats.get(
            "avg_boost",
            np.nan,
        ),
        "orange_amount_collected": orange_stats.get(
            "amount_collected",
            np.nan,
        ),
        "orange_amount_stolen": orange_stats.get(
            "amount_stolen",
            np.nan,
        ),
        "orange_total_distance": orange_stats.get(
            "total_distance",
            np.nan,
        ),
        "orange_time_ball_possession": orange_stats.get(
            "time_ball_possession",
            np.nan,
        ),
        "orange_time_in_front_ball": orange_stats.get(
            "time_in_front_ball",
            np.nan,
        ),
        "orange_time_behind_ball": orange_stats.get(
            "time_behind_ball",
            np.nan,
        ),
        "orange_time_offensive_half": orange_stats.get(
            "time_offensive_half",
            np.nan,
        ),
        "orange_time_defensive_half": orange_stats.get(
            "time_defensive_half",
            np.nan,
        ),
        "orange_demos_inflicted": orange_stats.get(
            "demos_inflicted",
            np.nan,
        ),
        "orange_demos_taken": orange_stats.get(
            "demos_taken",
            np.nan,
        ),
        "game_winner": game_winner,
    }

    return record


# ============================================================
# PROCESS EACH OPEN
# ============================================================

def process_open(open_name, output_file):

    print("\n" + "=" * 72)
    print(f"PROCESSING {open_name.upper()}")
    print("=" * 72)

    open_root = (
        BALLCHASING_ROOT
        / open_name
    )

    if not open_root.exists():
        raise FileNotFoundError(
            f"Open folder not found:\n{open_root}"
        )

    team_files = sorted(
        list(
            open_root.rglob("*-teams.csv")
        )
        +
        list(
            open_root.rglob("*_team.csv")
        )
    )

    print(
        f"\nTeam CSV files found: "
        f"{len(team_files)}"
    )

    records = []
    failures = []

    for i, team_file in enumerate(
        team_files,
        start=1,
    ):

        try:
            record = process_game(
                team_file,
                open_name,
                open_root,
            )

            records.append(record)

        except Exception as exc:
            failures.append(
                {
                    "file": str(team_file),
                    "error": repr(exc),
                }
            )

        if (
            i % 25 == 0
            or i == len(team_files)
        ):
            print(
                f"Processed {i}/{len(team_files)}"
            )

    games = pd.DataFrame(records)

    if len(games) > 0:
        games = games.sort_values(
            [
                "stage",
                "subgroup",
                "replay_id",
            ]
        ).reset_index(drop=True)

    games.to_csv(
        output_file,
        index=False,
    )

    print(
        f"\nGames successfully processed: "
        f"{len(games)}"
    )

    print(
        f"Processing failures: "
        f"{len(failures)}"
    )

    print(
        f"Columns created: "
        f"{len(games.columns)}"
    )

    if len(games) > 0:

        print(
            "\nGames by stage/group:"
        )

        stage_counts = (
            games
            .groupby(
                ["stage", "subgroup"],
                dropna=False,
            )
            .size()
        )

        print(stage_counts)

        print(
            "\nGame winners with known team names:"
        )

        print(
            games["game_winner"]
            .value_counts(
                dropna=False
            )
            .head(20)
        )

        missing_team_names = (
            games["blue_team"].isna()
            | games["orange_team"].isna()
        ).sum()

        missing_winner = (
            games["game_winner"].isna()
        ).sum()

        print(
            f"\nGames with at least one "
            f"missing team name: "
            f"{missing_team_names}"
        )

        print(
            f"Games with missing game_winner: "
            f"{missing_winner}"
        )

    if failures:
        print("\nFAILED FILES:")

        for failure in failures:
            print(
                failure["file"]
            )
            print(
                "  ",
                failure["error"],
            )

    print("\nSaved to:")
    print(output_file)

    return games, failures


# ============================================================
# MAIN
# ============================================================

print("=" * 72)
print("PROCESSING OPEN 2 AND OPEN 3 GAME EXPORTS")
print("=" * 72)

open2_games, open2_failures = process_open(
    "Open 2",
    OUTPUT_OPEN2,
)

open3_games, open3_failures = process_open(
    "Open 3",
    OUTPUT_OPEN3,
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 72)
print("FINAL SUMMARY")
print("=" * 72)

print(
    f"\nOpen 2 games: "
    f"{len(open2_games)}"
)

print(
    f"Open 3 games: "
    f"{len(open3_games)}"
)

print(
    f"Combined games: "
    f"{len(open2_games) + len(open3_games)}"
)

print(
    f"Open 2 failures: "
    f"{len(open2_failures)}"
)

print(
    f"Open 3 failures: "
    f"{len(open3_failures)}"
)

expected_open2 = 146
expected_open3 = 140

if (
    len(open2_games) == expected_open2
    and len(open3_games) == expected_open3
    and len(open2_failures) == 0
    and len(open3_failures) == 0
):
    print(
        "\nPASS: all 286 expected Open 2/Open 3 "
        "replays were processed."
    )
else:
    print(
        "\nWARNING: processed counts do not match "
        "the inventory totals."
    )

print("\nOutputs:")
print(OUTPUT_OPEN2)
print(OUTPUT_OPEN3)

print("\n" + "=" * 72)
print("DONE")
print("=" * 72)
