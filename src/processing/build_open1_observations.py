from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_GAMES = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ballchasing"
    / "open1_series_reconstructed.csv"
)

BLAST_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "blast"
    / "BLAST_Player_History.xlsx"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ballchasing"
)

OUTPUT_GAMES = OUTPUT_DIR / "open1_games_final.csv"
OUTPUT_SERIES = OUTPUT_DIR / "open1_series_final.csv"
OUTPUT_ML = OUTPUT_DIR / "open1_ml_observations.csv"
OUTPUT_UNMATCHED = OUTPUT_DIR / "open1_blast_unmatched_players.csv"


# ============================================================
# VERIFIED SERIES-LEVEL RESULT RECONCILIATIONS
#
# These records NEVER alter replay-derived game statistics.
# They only provide a verified completed-series result when
# the replay collection is incomplete.
# ============================================================

SERIES_RESULT_RECONCILIATIONS = {
    "O1_020": {
        "expected_team_a": "SELECAO",
        "expected_team_b": "SYNERGY",
        "completed_team_a_game_wins": 1,
        "completed_team_b_game_wins": 3,
        "completed_series_winner_side": "B",
        "result_source": "manually_reconciled_completed_series",
    },
}


# ============================================================
# PLAYER NAME HELPERS
# ============================================================

# Only aliases we explicitly know about.
ALIASES = {
    "bhavi": "ajg.",
    "ajg": "ajg.",
    "maykinhoo": "MayKo",
    "hyderr (new binds)": "Hyderr",
    "radosinho": "Radosin",
    "acro": "accro",
    "saizenrl": "saizen",
}

def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def split_players(value):
    """
    Convert a stored player string into a list of player names.

    Expected format is normally:
        Player1|Player2|Player3

    Some fallback separators are supported.
    """

    text = clean_text(value)

    if not text:
        return []

    if "|" in text:
        parts = text.split("|")
    elif ";" in text:
        parts = text.split(";")
    elif "," in text:
        parts = text.split(",")
    else:
        parts = [text]

    result = []

    for player in parts:
        player = player.strip()

        if not player:
            continue

        alias_key = player.lower()

        if alias_key in ALIASES:
            player = ALIASES[alias_key]

        result.append(player)

    return result


def canonical_player_name(name):
    """
    Conservative comparison form.

    This is NOT used to rename the player in the output.
    It is only used to compare identities between files.
    """

    name = clean_text(name)

    alias_key = name.lower()

    if alias_key in ALIASES:
        name = ALIASES[alias_key]

    return re.sub(r"[^a-z0-9]", "", name.lower())


def canonical_roster(players):
    """
    Order-independent roster fingerprint.
    """

    normalised = [
        canonical_player_name(player)
        for player in players
        if canonical_player_name(player)
    ]

    return tuple(sorted(normalised))


def display_roster(players):
    return "|".join(sorted(players, key=lambda x: x.lower()))


# ============================================================
# TEAM NAME HELPERS
# ============================================================

def valid_team_name(value):
    name = clean_text(value)

    if not name:
        return False

    if name.lower() in {
        "nan",
        "none",
        "null",
        "unknown",
        "unknown team",
        "unknown team names",
    }:
        return False

    return True


def most_common_team_name(names):
    cleaned = [
        clean_text(name)
        for name in names
        if valid_team_name(name)
    ]

    if not cleaned:
        return np.nan

    counts = pd.Series(cleaned).value_counts()

    return counts.index[0]


# ============================================================
# LOAD REPLAY DATA
# ============================================================

print("=" * 70)
print("BUILDING OPEN 1 SERIES AND ML OBSERVATIONS")
print("=" * 70)

if not INPUT_GAMES.exists():
    raise FileNotFoundError(
        f"Could not find:\n{INPUT_GAMES}"
    )

games = pd.read_csv(INPUT_GAMES)

print(f"\nReplay rows loaded: {len(games)}")


required_columns = [
    "replay_id",
    "stage",
    "blue_team",
    "orange_team",
    "blue_goals",
    "orange_goals",
    "replay_datetime_utc",
    "blue_players",
    "orange_players",
    "matchup_fingerprint",
]

missing_columns = [
    column
    for column in required_columns
    if column not in games.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# TIMESTAMPS
# ============================================================

games["replay_datetime_utc"] = pd.to_datetime(
    games["replay_datetime_utc"],
    utc=True,
    errors="coerce",
)

missing_times = games["replay_datetime_utc"].isna().sum()

print(f"Games with missing timestamps: {missing_times}")

if missing_times > 0:
    raise ValueError(
        "Some replay timestamps are missing. "
        "Stopping rather than guessing game order."
    )


# ============================================================
# BUILD CANONICAL PLAYER SIDES
# ============================================================

games["_blue_player_list"] = games["blue_players"].apply(split_players)
games["_orange_player_list"] = games["orange_players"].apply(split_players)

games["_blue_roster"] = games["_blue_player_list"].apply(canonical_roster)
games["_orange_roster"] = games["_orange_player_list"].apply(canonical_roster)


# ============================================================
# IMPORTANT:
# GROUP BY PLAYER MATCHUP ONLY
#
# We deliberately DO NOT include match_date here.
#
# If the same series crosses midnight / UTC date boundaries,
# it still remains one series.
# ============================================================

series_records = []
game_parts = []

fingerprints = (
    games["matchup_fingerprint"]
    .dropna()
    .unique()
)

print(f"Unique matchup fingerprints: {len(fingerprints)}")


# Sort fingerprints by the first replay time so series IDs are
# chronological rather than arbitrary.
fingerprint_first_times = (
    games.groupby("matchup_fingerprint")["replay_datetime_utc"]
    .min()
    .sort_values()
)

ordered_fingerprints = fingerprint_first_times.index.tolist()


for series_number, fingerprint in enumerate(
    ordered_fingerprints,
    start=1,
):

    group = games[
        games["matchup_fingerprint"] == fingerprint
    ].copy()

    group = group.sort_values(
        ["replay_datetime_utc", "replay_id"]
    ).reset_index(drop=True)

    series_id = f"O1_{series_number:03d}"

    # --------------------------------------------------------
    # Determine the two canonical roster sides.
    #
    # Use the first replay as the orientation anchor.
    # This does NOT mean blue = Team A forever.
    # It simply creates a stable A/B identity for the series.
    # --------------------------------------------------------

    first_row = group.iloc[0]

    team_a_roster = first_row["_blue_roster"]
    team_b_roster = first_row["_orange_roster"]

    team_a_player_names = first_row["_blue_player_list"]
    team_b_player_names = first_row["_orange_player_list"]

    team_a_names_seen = []
    team_b_names_seen = []

    game_winner_sides = []
    orientation_errors = []

    side_a_goals_total = 0
    side_b_goals_total = 0

    for index, row in group.iterrows():

        blue_roster = row["_blue_roster"]
        orange_roster = row["_orange_roster"]

        # ----------------------------------------------------
        # Work out whether Team A is blue or orange
        # for this individual replay.
        # ----------------------------------------------------

        if (
            blue_roster == team_a_roster
            and orange_roster == team_b_roster
        ):
            team_a_colour = "blue"

        elif (
            blue_roster == team_b_roster
            and orange_roster == team_a_roster
        ):
            team_a_colour = "orange"

        else:
            team_a_colour = "unresolved"

        orientation_errors.append(
            team_a_colour == "unresolved"
        )

        # ----------------------------------------------------
        # Retain team labels where Ballchasing supplied them.
        # Player identities are NOT being used to invent team
        # names.
        # ----------------------------------------------------

        if team_a_colour == "blue":

            if valid_team_name(row["blue_team"]):
                team_a_names_seen.append(row["blue_team"])

            if valid_team_name(row["orange_team"]):
                team_b_names_seen.append(row["orange_team"])

            a_goals = row["blue_goals"]
            b_goals = row["orange_goals"]

        elif team_a_colour == "orange":

            if valid_team_name(row["orange_team"]):
                team_a_names_seen.append(row["orange_team"])

            if valid_team_name(row["blue_team"]):
                team_b_names_seen.append(row["blue_team"])

            a_goals = row["orange_goals"]
            b_goals = row["blue_goals"]

        else:
            a_goals = np.nan
            b_goals = np.nan

        # ----------------------------------------------------
        # Determine winner from GOALS, not score/performance.
        # ----------------------------------------------------

        if pd.notna(a_goals) and pd.notna(b_goals):

            side_a_goals_total += a_goals
            side_b_goals_total += b_goals

            if a_goals > b_goals:
                winner_side = "A"

            elif b_goals > a_goals:
                winner_side = "B"

            else:
                winner_side = np.nan

        else:
            winner_side = np.nan

        game_winner_sides.append(winner_side)

    group["series_id"] = series_id
    group["game_number_in_series"] = (
        np.arange(len(group)) + 1
    )

    group["team_a_players"] = display_roster(
        team_a_player_names
    )

    group["team_b_players"] = display_roster(
        team_b_player_names
    )

    group["game_winner_side"] = game_winner_sides

    group["team_a_colour"] = [
        (
            "blue"
            if row["_blue_roster"] == team_a_roster
            else
            "orange"
            if row["_orange_roster"] == team_a_roster
            else
            "unresolved"
        )
        for _, row in group.iterrows()
    ]

    team_a_name = most_common_team_name(team_a_names_seen)
    team_b_name = most_common_team_name(team_b_names_seen)

    group["team_a"] = team_a_name
    group["team_b"] = team_b_name

    # --------------------------------------------------------
    # SERIES RESULT
    # --------------------------------------------------------

    team_a_game_wins = sum(
        winner == "A"
        for winner in game_winner_sides
    )

    team_b_game_wins = sum(
        winner == "B"
        for winner in game_winner_sides
    )

    if team_a_game_wins > team_b_game_wins:
        series_winner_side = "A"

    elif team_b_game_wins > team_a_game_wins:
        series_winner_side = "B"

    else:
        series_winner_side = np.nan

    if series_winner_side == "A":
        series_winner = team_a_name

    elif series_winner_side == "B":
        series_winner = team_b_name

    else:
        series_winner = np.nan

    if series_winner_side == "A":
        target_team_a_win = 1

    elif series_winner_side == "B":
        target_team_a_win = 0

    else:
        target_team_a_win = np.nan

    # --------------------------------------------------------
    # STAGE INFORMATION
    # --------------------------------------------------------

    stages = [
        clean_text(x)
        for x in group["stage"].dropna().unique()
    ]

    subgroups = []

    if "subgroup" in group.columns:
        subgroups = [
            clean_text(x)
            for x in group["subgroup"].dropna().unique()
            if clean_text(x)
        ]

    stage = stages[0] if stages else np.nan

    subgroup = (
        subgroups[0]
        if len(subgroups) == 1
        else np.nan
    )

    # Group stage = BO5, playoffs = BO7.
    # This is only used as a diagnostic.
    if "group" in clean_text(stage).lower():
        wins_required = 3
    else:
        wins_required = 4

    winner_reaches_required_wins = (
        max(team_a_game_wins, team_b_game_wins)
        >= wins_required
    )

    series_record = {
        "series_id": series_id,
        "open": "Open 1",
        "stage": stage,
        "subgroup": subgroup,
        "series_start_utc": group[
            "replay_datetime_utc"
        ].min(),
        "series_end_utc": group[
            "replay_datetime_utc"
        ].max(),
        "team_a": team_a_name,
        "team_b": team_b_name,
        "team_a_players": display_roster(
            team_a_player_names
        ),
        "team_b_players": display_roster(
            team_b_player_names
        ),
        "replay_count": len(group),
        "team_a_game_wins_observed": team_a_game_wins,
        "team_b_game_wins_observed": team_b_game_wins,
        "series_winner_side": series_winner_side,
        "series_winner": series_winner,
        "target_team_a_win": target_team_a_win,
        "team_a_goals_observed": side_a_goals_total,
        "team_b_goals_observed": side_b_goals_total,
        "wins_required_for_stage": wins_required,
        "winner_reaches_required_wins": (
            winner_reaches_required_wins
        ),
        "orientation_error": any(
            orientation_errors
        ),
        "matchup_fingerprint": fingerprint,
        "replay_ids": "|".join(
            group["replay_id"].astype(str)
        ),
    }

    series_records.append(series_record)
    game_parts.append(group)


# ============================================================
# CREATE FINAL GAME FILE
# ============================================================

games_final = pd.concat(
    game_parts,
    ignore_index=True,
)

# Drop internal helper columns.
games_final = games_final.drop(
    columns=[
        "_blue_player_list",
        "_orange_player_list",
        "_blue_roster",
        "_orange_roster",
    ],
    errors="ignore",
)

games_final = games_final.sort_values(
    [
        "series_id",
        "game_number_in_series",
    ]
).reset_index(drop=True)


# ============================================================
# CREATE SERIES FILE
# ============================================================

series = pd.DataFrame(series_records)

series = series.sort_values(
    "series_start_utc"
).reset_index(drop=True)


# ============================================================
# RECONCILE VERIFIED COMPLETED-SERIES RESULTS
#
# The replay-derived fields remain untouched:
#   replay_count
#   team_a_game_wins_observed
#   team_b_game_wins_observed
#   team_a_goals_observed
#   team_b_goals_observed
#   winner_reaches_required_wins
#
# Only completed-series label metadata and the supervised
# target are reconciled where we have a verified result.
# ============================================================

series["result_reconciled"] = False
series["result_source"] = "observed_replays"
series["completed_team_a_game_wins"] = np.nan
series["completed_team_b_game_wins"] = np.nan
series["completed_series_winner_side"] = pd.Series(pd.NA, index=series.index, dtype="string")

for series_id, reconciliation in SERIES_RESULT_RECONCILIATIONS.items():

    mask = series["series_id"] == series_id

    if not mask.any():
        raise ValueError(
            f"Reconciliation requested for {series_id}, "
            "but that series_id was not found."
        )

    if mask.sum() != 1:
        raise ValueError(
            f"Expected exactly one row for {series_id}, "
            f"found {int(mask.sum())}."
        )

    current_team_a = clean_text(
        series.loc[mask, "team_a"].iloc[0]
    )
    current_team_b = clean_text(
        series.loc[mask, "team_b"].iloc[0]
    )

    expected_team_a = reconciliation[
        "expected_team_a"
    ]
    expected_team_b = reconciliation[
        "expected_team_b"
    ]

    if (
        current_team_a != expected_team_a
        or current_team_b != expected_team_b
    ):
        raise ValueError(
            f"{series_id} reconciliation identity check failed. "
            f"Expected {expected_team_a} vs {expected_team_b}, "
            f"but found {current_team_a} vs {current_team_b}."
        )

    completed_a_wins = reconciliation[
        "completed_team_a_game_wins"
    ]
    completed_b_wins = reconciliation[
        "completed_team_b_game_wins"
    ]
    completed_winner_side = reconciliation[
        "completed_series_winner_side"
    ]

    if completed_winner_side not in {"A", "B"}:
        raise ValueError(
            f"Invalid completed winner side for {series_id}: "
            f"{completed_winner_side}"
        )

    if completed_winner_side == "A":
        completed_series_winner = current_team_a
        completed_target = 1
    else:
        completed_series_winner = current_team_b
        completed_target = 0

    series.loc[
        mask,
        "completed_team_a_game_wins"
    ] = completed_a_wins

    series.loc[
        mask,
        "completed_team_b_game_wins"
    ] = completed_b_wins

    series.loc[
        mask,
        "completed_series_winner_side"
    ] = completed_winner_side

    series.loc[
        mask,
        "series_winner_side"
    ] = completed_winner_side

    series.loc[
        mask,
        "series_winner"
    ] = completed_series_winner

    series.loc[
        mask,
        "target_team_a_win"
    ] = completed_target

    series.loc[
        mask,
        "result_reconciled"
    ] = True

    series.loc[
        mask,
        "result_source"
    ] = reconciliation["result_source"]


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("SERIES RECONSTRUCTION")
print("=" * 70)

print(f"Games retained: {len(games_final)}")
print(f"Series created: {len(series)}")

print("\nGames per series:")
print(
    series["replay_count"]
    .value_counts()
    .sort_index()
)

print(
    "\nSeries with orientation errors:",
    int(series["orientation_error"].sum()),
)

print(
    "Series where observed winner does not reach "
    "the expected win threshold:",
    int(
        (~series["winner_reaches_required_wins"])
        .sum()
    ),
)

unusual = series[
    (series["replay_count"] < 3)
    | (series["replay_count"] > 7)
    | (series["orientation_error"])
    | (~series["winner_reaches_required_wins"])
]

if len(unusual) > 0:

    print("\nSeries requiring inspection:")

    print(
        unusual[
            [
                "series_id",
                "stage",
                "team_a",
                "team_b",
                "team_a_players",
                "team_b_players",
                "replay_count",
                "team_a_game_wins_observed",
                "team_b_game_wins_observed",
                "winner_reaches_required_wins",
                "orientation_error",
            ]
        ].to_string(index=False)
    )

else:
    print("\nNo unusual series detected.")


# ============================================================
# SAVE CLEAN GAME AND SERIES DATA
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

games_final.to_csv(
    OUTPUT_GAMES,
    index=False,
)

series.to_csv(
    OUTPUT_SERIES,
    index=False,
)

print(f"\nSaved game-level data:\n{OUTPUT_GAMES}")
print(f"\nSaved series-level data:\n{OUTPUT_SERIES}")


# ============================================================
# BUILD INITIAL ML OBSERVATIONS FROM BLAST HISTORY
#
# BLAST values are historical through end-2024, so they are
# safe pre-Open-1 features.
# ============================================================

if not BLAST_FILE.exists():

    print(
        "\nBLAST file was not found, so ML feature "
        "construction was skipped:"
    )

    print(BLAST_FILE)

else:

    print("\n" + "=" * 70)
    print("ADDING BLAST HISTORICAL FEATURES")
    print("=" * 70)

    blast = pd.read_excel(
        BLAST_FILE,
        sheet_name="Sheet1",
    )

    if "player_name" not in blast.columns:
        raise ValueError(
            "BLAST sheet does not contain player_name."
        )

    blast["_match_name"] = blast[
        "player_name"
    ].apply(canonical_player_name)

    # Prevent accidental fuzzy matching.
    blast_lookup = {}

    duplicate_keys = set()

    for _, row in blast.iterrows():

        key = row["_match_name"]

        if not key:
            continue

        if key in blast_lookup:
            duplicate_keys.add(key)

        else:
            blast_lookup[key] = row

    if duplicate_keys:
        print(
            "\nWarning: duplicate normalised BLAST names:"
        )
        print(sorted(duplicate_keys))

    historical_features = [
        "career_games",
        "career_win_pct",
        "career_score_avg",
        "career_goals_avg",
        "career_assists_avg",
        "career_shots_avg",
        "career_saves_avg",
        "career_demos_avg",
        "career_rating",
        "2024_games",
        "2024_win_pct",
        "2024_score_avg",
        "2024_goals_avg",
        "2024_assists_avg",
        "2024_shots_avg",
        "2024_saves_avg",
        "2024_demos_avg",
        "2024_rating",
    ]

    available_features = [
        feature
        for feature in historical_features
        if feature in blast.columns
    ]

    unmatched_records = []
    ml_rows = []

    for _, row in series.iterrows():

        ml_row = {
            "series_id": row["series_id"],
            "open": row["open"],
            "stage": row["stage"],
            "subgroup": row["subgroup"],
            "series_start_utc": row[
                "series_start_utc"
            ],
            "team_a": row["team_a"],
            "team_b": row["team_b"],
            "team_a_players": row[
                "team_a_players"
            ],
            "team_b_players": row[
                "team_b_players"
            ],
            "target_team_a_win": row[
                "target_team_a_win"
            ],
            "winner_reaches_required_wins": row[
                "winner_reaches_required_wins"
            ],
            "result_reconciled": row[
                "result_reconciled"
            ],
            "result_source": row[
                "result_source"
            ],
            "completed_team_a_game_wins": row[
                "completed_team_a_game_wins"
            ],
            "completed_team_b_game_wins": row[
                "completed_team_b_game_wins"
            ],
            "completed_series_winner_side": row[
                "completed_series_winner_side"
            ],
        }

        side_feature_values = {}

        for side in ["a", "b"]:

            players = split_players(
                row[f"team_{side}_players"]
            )

            player_rows = []

            for player in players:

                key = canonical_player_name(
                    player
                )

                blast_row = blast_lookup.get(key)

                if blast_row is None:

                    unmatched_records.append(
                        {
                            "series_id": row[
                                "series_id"
                            ],
                            "side": side.upper(),
                            "player_name": player,
                            "normalised_name": key,
                        }
                    )

                    continue

                player_rows.append(blast_row)

            side_feature_values[side] = {}

            for feature in available_features:

                values = []

                for blast_row in player_rows:

                    value = pd.to_numeric(
                        blast_row[feature],
                        errors="coerce",
                    )

                    if pd.notna(value):
                        values.append(float(value))

                if values:
                    mean_value = np.mean(values)
                else:
                    mean_value = np.nan

                output_name = (
                    f"team_{side}_{feature}_mean"
                )

                ml_row[output_name] = mean_value

                side_feature_values[side][
                    feature
                ] = mean_value

            # ------------------------------------------------
            # BLAST coverage diagnostics
            #
            # A player can have a successfully matched BLAST row
            # while still having no usable historical numeric
            # statistics. Keep those concepts separate.
            # ------------------------------------------------

            ml_row[
                f"team_{side}_blast_players_total"
            ] = len(players)

            ml_row[
                f"team_{side}_blast_players_name_matched"
            ] = len(player_rows)

            players_with_stats = 0

            for blast_row in player_rows:

                has_any_stat = False

                for feature in available_features:

                    value = pd.to_numeric(
                        blast_row[feature],
                        errors="coerce",
                    )

                    if pd.notna(value):
                        has_any_stat = True
                        break

                if has_any_stat:
                    players_with_stats += 1

            ml_row[
                f"team_{side}_blast_players_with_stats"
            ] = players_with_stats

            if len(players) > 0:
                coverage = (
                    players_with_stats / len(players)
                )
            else:
                coverage = np.nan

            ml_row[
                f"team_{side}_blast_stats_coverage"
            ] = coverage

        # ----------------------------------------------------
        # Difference features:
        # Team A minus Team B
        # ----------------------------------------------------

        for feature in available_features:

            a_value = side_feature_values[
                "a"
            ].get(feature, np.nan)

            b_value = side_feature_values[
                "b"
            ].get(feature, np.nan)

            if (
                pd.notna(a_value)
                and pd.notna(b_value)
            ):
                difference = a_value - b_value
            else:
                difference = np.nan

            ml_row[
                f"diff_{feature}"
            ] = difference

        ml_rows.append(ml_row)

    ml = pd.DataFrame(ml_rows)

    # Keep only rows with a known target.
    ml["target_team_a_win"] = pd.to_numeric(
        ml["target_team_a_win"],
        errors="coerce",
    )

    ml.to_csv(
        OUTPUT_ML,
        index=False,
    )

    unmatched = pd.DataFrame(
        unmatched_records
    ).drop_duplicates()

    unmatched.to_csv(
        OUTPUT_UNMATCHED,
        index=False,
    )

    print(f"\nML observations created: {len(ml)}")

    if len(unmatched) == 0:

        print(
            "All roster names matched to BLAST history."
        )

    else:

        print(
            "Unique unmatched BLAST player names:",
            unmatched["player_name"].nunique(),
        )

        print("\nUnmatched names:")
        print(
            unmatched[
                "player_name"
            ].drop_duplicates().to_string(
                index=False
            )
        )

    print(f"\nSaved ML observations:\n{OUTPUT_ML}")

    print(
        f"\nSaved BLAST matching report:\n"
        f"{OUTPUT_UNMATCHED}"
    )


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)