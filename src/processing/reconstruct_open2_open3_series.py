from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ballchasing"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ballchasing"
)

OPEN_CONFIG = {
    "Open 2": {
        "prefix": "O2",
        "games_file": PROCESSED_DIR / "open2_games.csv",
        "metadata_file": PROCESSED_DIR / "open2_replay_metadata.csv",
        "games_output": PROCESSED_DIR / "open2_games_reconstructed.csv",
        "series_output": PROCESSED_DIR / "open2_series_reconstructed.csv",
        "matchups_output": PROCESSED_DIR / "open2_player_matchups.csv",
    },
    "Open 3": {
        "prefix": "O3",
        "games_file": PROCESSED_DIR / "open3_games.csv",
        "metadata_file": PROCESSED_DIR / "open3_replay_metadata.csv",
        "games_output": PROCESSED_DIR / "open3_games_reconstructed.csv",
        "series_output": PROCESSED_DIR / "open3_series_reconstructed.csv",
        "matchups_output": PROCESSED_DIR / "open3_player_matchups.csv",
    },
}


# ============================================================
# KNOWN PLAYER-NAME ALIASES
# ============================================================

# These are username normalisations only.
# They are NOT used to infer team names.
PLAYER_ALIASES = {
    "MayKinhoo": "MayKo",
    "Hyderr (new binds)": "Hyderr",
    "Radosinho": "Radosin",
    "acro": "accro",
    "saizenrl": "saizen",
    "AJG": "ajg.",
    "bhavi": "ajg.",
}


# ============================================================
# HELPERS
# ============================================================

def divider(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def clean_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def normalise_player_name(name):
    """
    Apply only known aliases.
    Do not do fuzzy matching here.
    """
    name = clean_text(name)

    if name in PLAYER_ALIASES:
        return PLAYER_ALIASES[name]

    return name


def read_ballchasing_csv(path):
    return pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
    )


def first_existing_column(df, candidates):
    lower_map = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in lower_map:
            return lower_map[key]

    return None


def expected_wins_for_stage(stage):
    """
    Group Stage = BO5 => first to 3.
    Playoffs = BO7 => first to 4.
    """
    stage_text = clean_text(stage).lower()

    if "group" in stage_text:
        return 3

    return 4


def bool_from_value(value):
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


# ============================================================
# PLAYER FILE INDEX
# ============================================================

def replay_id_from_player_file(path):
    name = path.name

    if name.lower().endswith("-players.csv"):
        return name[:-12]

    if name.lower().endswith("_player.csv"):
        return name[:-11]

    return None


def build_player_file_index(open_root):
    """
    Index every player CSV by replay ID.
    """

    player_files = (
        list(open_root.rglob("*-players.csv"))
        +
        list(open_root.rglob("*_player.csv"))
    )

    index = {}

    duplicate_ids = []

    for path in player_files:

        replay_id = replay_id_from_player_file(path)

        if replay_id is None:
            continue

        if replay_id in index:
            duplicate_ids.append(replay_id)

        index[replay_id] = path

    return index, duplicate_ids


# ============================================================
# PLAYER SIDE EXTRACTION
# ============================================================

def extract_player_sides(player_file):
    """
    Return normalised player names on blue and orange.

    We deliberately do NOT reject 4v4 or other unusual player
    counts. The replay remains valid.
    """

    df = read_ballchasing_csv(player_file)

    color_col = first_existing_column(
        df,
        ["color", "team color"],
    )

    player_col = first_existing_column(
        df,
        ["player name", "name", "player"],
    )

    if color_col is None:
        raise ValueError(
            f"No color column in {player_file}"
        )

    if player_col is None:
        raise ValueError(
            f"No player-name column in {player_file}"
        )

    colors = (
        df[color_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    blue_names = [
        normalise_player_name(name)
        for name in df.loc[
            colors == "blue",
            player_col,
        ].tolist()
        if clean_text(name) != ""
    ]

    orange_names = [
        normalise_player_name(name)
        for name in df.loc[
            colors == "orange",
            player_col,
        ].tolist()
        if clean_text(name) != ""
    ]

    blue_names = sorted(
        dict.fromkeys(blue_names)
    )

    orange_names = sorted(
        dict.fromkeys(orange_names)
    )

    blue_signature = "|".join(
        blue_names
    )

    orange_signature = "|".join(
        orange_names
    )

    if blue_signature <= orange_signature:
        matchup_fingerprint = (
            f"{blue_signature} || "
            f"{orange_signature}"
        )
    else:
        matchup_fingerprint = (
            f"{orange_signature} || "
            f"{blue_signature}"
        )

    return {
        "blue_players": blue_signature,
        "orange_players": orange_signature,
        "blue_player_count": len(blue_names),
        "orange_player_count": len(orange_names),
        "matchup_fingerprint": matchup_fingerprint,
    }


# ============================================================
# GAME SIDE ORIENTATION
# ============================================================

def canonical_sides_from_fingerprint(fingerprint):
    parts = str(fingerprint).split(" || ")

    if len(parts) != 2:
        raise ValueError(
            f"Invalid matchup fingerprint: {fingerprint}"
        )

    return parts[0], parts[1]


def orient_game(row):
    """
    Canonical side A/B is based only on the sorted player-side
    signatures in matchup_fingerprint.

    Team names remain the existing replay team names.
    """

    side_a_players, side_b_players = (
        canonical_sides_from_fingerprint(
            row["matchup_fingerprint"]
        )
    )

    blue_players = clean_text(
        row["blue_players"]
    )

    orange_players = clean_text(
        row["orange_players"]
    )

    if (
        blue_players == side_a_players
        and orange_players == side_b_players
    ):
        team_a = row["blue_team"]
        team_b = row["orange_team"]

        team_a_goals = row["blue_goals"]
        team_b_goals = row["orange_goals"]

        blue_side = "A"
        orange_side = "B"

    elif (
        blue_players == side_b_players
        and orange_players == side_a_players
    ):
        team_a = row["orange_team"]
        team_b = row["blue_team"]

        team_a_goals = row["orange_goals"]
        team_b_goals = row["blue_goals"]

        blue_side = "B"
        orange_side = "A"

    else:
        return {
            "orientation_error": True,
            "team_a_players": side_a_players,
            "team_b_players": side_b_players,
            "team_a": np.nan,
            "team_b": np.nan,
            "team_a_goals": np.nan,
            "team_b_goals": np.nan,
            "blue_side": np.nan,
            "orange_side": np.nan,
            "game_winner_side": np.nan,
        }

    team_a_goals = pd.to_numeric(
        team_a_goals,
        errors="coerce",
    )

    team_b_goals = pd.to_numeric(
        team_b_goals,
        errors="coerce",
    )

    if (
        pd.notna(team_a_goals)
        and pd.notna(team_b_goals)
    ):
        if team_a_goals > team_b_goals:
            winner_side = "A"

        elif team_b_goals > team_a_goals:
            winner_side = "B"

        else:
            winner_side = np.nan

    else:
        winner_side = np.nan

    return {
        "orientation_error": False,
        "team_a_players": side_a_players,
        "team_b_players": side_b_players,
        "team_a": team_a,
        "team_b": team_b,
        "team_a_goals": team_a_goals,
        "team_b_goals": team_b_goals,
        "blue_side": blue_side,
        "orange_side": orange_side,
        "game_winner_side": winner_side,
    }


# ============================================================
# SERIES SPLITTING
# ============================================================

def split_matchup_into_series(group):
    """
    Split one same-roster/stage/subgroup matchup into one or more
    completed series.

    Games are sorted strictly by trusted replay timestamp.

    A series closes immediately when either side reaches the
    required number of game wins:
      Group Stage -> 3
      Playoffs    -> 4

    This prevents accidental merging if the same two rosters meet
    more than once in the same Open.
    """

    group = group.sort_values(
        [
            "replay_timestamp_utc",
            "replay_id",
        ]
    ).copy()

    completed_chunks = []

    current_indices = []
    a_wins = 0
    b_wins = 0

    for idx, row in group.iterrows():

        current_indices.append(idx)

        winner_side = row[
            "game_winner_side"
        ]

        if winner_side == "A":
            a_wins += 1

        elif winner_side == "B":
            b_wins += 1

        wins_required = expected_wins_for_stage(
            row["stage"]
        )

        if (
            a_wins >= wins_required
            or b_wins >= wins_required
        ):
            completed_chunks.append(
                current_indices
            )

            current_indices = []
            a_wins = 0
            b_wins = 0

    if current_indices:
        completed_chunks.append(
            current_indices
        )

    return completed_chunks


# ============================================================
# PROCESS ONE OPEN
# ============================================================

def process_open(open_name, config):

    divider(
        f"RECONSTRUCTING {open_name.upper()} SERIES"
    )

    games_file = config[
        "games_file"
    ]

    metadata_file = config[
        "metadata_file"
    ]

    if not games_file.exists():
        raise FileNotFoundError(
            f"Missing games file:\n{games_file}"
        )

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Missing metadata file:\n{metadata_file}"
        )

    games = pd.read_csv(
        games_file
    )

    metadata = pd.read_csv(
        metadata_file
    )

    print(
        f"\nGames loaded: "
        f"{len(games)}"
    )

    # --------------------------------------------------------
    # Timestamp merge
    # --------------------------------------------------------

    metadata_keep = metadata[
        [
            "replay_id",
            "replay_timestamp_utc",
        ]
    ].copy()

    games = games.merge(
        metadata_keep,
        on="replay_id",
        how="left",
        validate="one_to_one",
    )

    games[
        "replay_timestamp_utc"
    ] = pd.to_datetime(
        games[
            "replay_timestamp_utc"
        ],
        errors="coerce",
        utc=True,
    )

    missing_timestamps = int(
        games[
            "replay_timestamp_utc"
        ].isna().sum()
    )

    print(
        f"Games with missing timestamps: "
        f"{missing_timestamps}"
    )

    # --------------------------------------------------------
    # Player-file index
    # --------------------------------------------------------

    open_root = (
        RAW_ROOT
        / open_name
    )

    player_index, duplicate_ids = (
        build_player_file_index(
            open_root
        )
    )

    print(
        f"Player CSV files found: "
        f"{len(player_index)}"
    )

    if duplicate_ids:
        print(
            "WARNING: duplicate replay IDs in "
            "player-file index:",
            len(duplicate_ids),
        )

    # --------------------------------------------------------
    # Extract player sides
    # --------------------------------------------------------

    player_records = []
    player_failures = []

    for replay_id in games[
        "replay_id"
    ].astype(str):

        player_file = player_index.get(
            replay_id
        )

        if player_file is None:
            player_failures.append(
                {
                    "replay_id": replay_id,
                    "error": "player_file_not_found",
                }
            )
            continue

        try:
            info = extract_player_sides(
                player_file
            )

            info[
                "replay_id"
            ] = replay_id

            info[
                "source_player_file"
            ] = str(
                player_file.relative_to(
                    PROJECT_ROOT
                )
            )

            player_records.append(
                info
            )

        except Exception as exc:
            player_failures.append(
                {
                    "replay_id": replay_id,
                    "error": repr(exc),
                }
            )

    players = pd.DataFrame(
        player_records
    )

    print(
        f"Player files successfully processed: "
        f"{len(players)}"
    )

    print(
        f"Player-file failures: "
        f"{len(player_failures)}"
    )

    games = games.merge(
        players,
        on="replay_id",
        how="left",
        validate="one_to_one",
    )

    missing_fingerprints = int(
        games[
            "matchup_fingerprint"
        ].isna().sum()
    )

    print(
        f"Games with player matchup fingerprints: "
        f"{len(games) - missing_fingerprints}"
    )

    print(
        f"Games without player matchup fingerprints: "
        f"{missing_fingerprints}"
    )

    # --------------------------------------------------------
    # Canonical side orientation
    # --------------------------------------------------------

    orientation_rows = []

    for _, row in games.iterrows():
        orientation_rows.append(
            orient_game(row)
        )

    orientation_df = pd.DataFrame(
        orientation_rows
    )

    for col in orientation_df.columns:
        games[col] = orientation_df[col].values

    orientation_errors = int(
        games[
            "orientation_error"
        ].fillna(True)
        .sum()
    )

    print(
        f"Games with orientation errors: "
        f"{orientation_errors}"
    )

    # --------------------------------------------------------
    # Candidate grouping
    #
    # Fingerprint + stage + subgroup keeps known tournament
    # phases separate. Within each group, completed-series
    # thresholds split rematches safely.
    # --------------------------------------------------------

    group_cols = [
        "matchup_fingerprint",
        "stage",
        "subgroup",
    ]

    provisional_series = []

    group_number = 0

    for _, group in games.groupby(
        group_cols,
        dropna=False,
        sort=False,
    ):

        chunks = split_matchup_into_series(
            group
        )

        for chunk_number, indices in enumerate(
            chunks,
            start=1,
        ):
            group_number += 1

            provisional_key = (
                f"candidate_{group_number:03d}"
            )

            for idx in indices:
                provisional_series.append(
                    {
                        "index": idx,
                        "provisional_series_key": (
                            provisional_key
                        ),
                    }
                )

    mapping = pd.DataFrame(
        provisional_series
    ).set_index("index")

    games[
        "provisional_series_key"
    ] = mapping[
        "provisional_series_key"
    ]

    # --------------------------------------------------------
    # Build one row per provisional series
    # --------------------------------------------------------

    series_records = []

    for provisional_key, group in games.groupby(
        "provisional_series_key",
        sort=False,
    ):

        group = group.sort_values(
            [
                "replay_timestamp_utc",
                "replay_id",
            ]
        ).copy()

        first = group.iloc[0]

        team_a_wins = int(
            (
                group[
                    "game_winner_side"
                ]
                == "A"
            ).sum()
        )

        team_b_wins = int(
            (
                group[
                    "game_winner_side"
                ]
                == "B"
            ).sum()
        )

        wins_required = (
            expected_wins_for_stage(
                first["stage"]
            )
        )

        if (
            team_a_wins >= wins_required
            and team_a_wins > team_b_wins
        ):
            winner_side = "A"

        elif (
            team_b_wins >= wins_required
            and team_b_wins > team_a_wins
        ):
            winner_side = "B"

        else:
            winner_side = np.nan

        if winner_side == "A":
            series_winner = first[
                "team_a"
            ]
            target_team_a_win = 1

        elif winner_side == "B":
            series_winner = first[
                "team_b"
            ]
            target_team_a_win = 0

        else:
            series_winner = np.nan
            target_team_a_win = np.nan

        winner_reaches_required = bool(
            (
                team_a_wins >= wins_required
                and team_a_wins > team_b_wins
            )
            or
            (
                team_b_wins >= wins_required
                and team_b_wins > team_a_wins
            )
        )

        team_a_names = (
            group["team_a"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        team_b_names = (
            group["team_b"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        team_a_unique = sorted(
            set(
                name
                for name in team_a_names
                if name != ""
            )
        )

        team_b_unique = sorted(
            set(
                name
                for name in team_b_names
                if name != ""
            )
        )

        team_name_consistency = (
            len(team_a_unique) <= 1
            and len(team_b_unique) <= 1
        )

        team_a = (
            team_a_unique[0]
            if len(team_a_unique) == 1
            else (
                first["team_a"]
                if pd.notna(
                    first["team_a"]
                )
                else np.nan
            )
        )

        team_b = (
            team_b_unique[0]
            if len(team_b_unique) == 1
            else (
                first["team_b"]
                if pd.notna(
                    first["team_b"]
                )
                else np.nan
            )
        )

        series_records.append(
            {
                "provisional_series_key": provisional_key,
                "open": open_name,
                "stage": first["stage"],
                "subgroup": first["subgroup"],
                "matchup_fingerprint": first[
                    "matchup_fingerprint"
                ],
                "team_a": team_a,
                "team_b": team_b,
                "team_a_players": first[
                    "team_a_players"
                ],
                "team_b_players": first[
                    "team_b_players"
                ],
                "series_start_utc": group[
                    "replay_timestamp_utc"
                ].min(),
                "series_end_utc": group[
                    "replay_timestamp_utc"
                ].max(),
                "replay_count": len(group),
                "team_a_game_wins_observed": (
                    team_a_wins
                ),
                "team_b_game_wins_observed": (
                    team_b_wins
                ),
                "wins_required": wins_required,
                "winner_reaches_required_wins": (
                    winner_reaches_required
                ),
                "series_winner_side": (
                    winner_side
                ),
                "series_winner": (
                    series_winner
                ),
                "target_team_a_win": (
                    target_team_a_win
                ),
                "orientation_error": bool(
                    group[
                        "orientation_error"
                    ].fillna(True).any()
                ),
                "team_name_consistency": (
                    team_name_consistency
                ),
            }
        )

    series = pd.DataFrame(
        series_records
    )

    # --------------------------------------------------------
    # Chronological series IDs
    # --------------------------------------------------------

    series = series.sort_values(
        [
            "series_start_utc",
            "provisional_series_key",
        ]
    ).reset_index(drop=True)

    series[
        "series_id"
    ] = [
        f"{config['prefix']}_{i:03d}"
        for i in range(
            1,
            len(series) + 1,
        )
    ]

    series_id_map = dict(
        zip(
            series[
                "provisional_series_key"
            ],
            series[
                "series_id"
            ],
        )
    )

    games[
        "series_id"
    ] = games[
        "provisional_series_key"
    ].map(
        series_id_map
    )

    games = games.sort_values(
        [
            "replay_timestamp_utc",
            "replay_id",
        ]
    ).reset_index(drop=True)

    series = series[
        [
            "series_id",
            "open",
            "stage",
            "subgroup",
            "matchup_fingerprint",
            "team_a",
            "team_b",
            "team_a_players",
            "team_b_players",
            "series_start_utc",
            "series_end_utc",
            "replay_count",
            "team_a_game_wins_observed",
            "team_b_game_wins_observed",
            "wins_required",
            "winner_reaches_required_wins",
            "series_winner_side",
            "series_winner",
            "target_team_a_win",
            "orientation_error",
            "team_name_consistency",
            "provisional_series_key",
        ]
    ]

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    games.to_csv(
        config[
            "games_output"
        ],
        index=False,
    )

    series.to_csv(
        config[
            "series_output"
        ],
        index=False,
    )

    matchup_cols = [
        "replay_id",
        "series_id",
        "replay_timestamp_utc",
        "stage",
        "subgroup",
        "blue_team",
        "orange_team",
        "blue_players",
        "orange_players",
        "blue_player_count",
        "orange_player_count",
        "matchup_fingerprint",
        "team_a_players",
        "team_b_players",
        "team_a",
        "team_b",
        "game_winner_side",
    ]

    games[
        matchup_cols
    ].to_csv(
        config[
            "matchups_output"
        ],
        index=False,
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    divider(
        f"{open_name.upper()} RECONSTRUCTION SUMMARY"
    )

    print(
        f"Games retained: "
        f"{len(games)}"
    )

    print(
        f"Series created: "
        f"{len(series)}"
    )

    print(
        "\nGames per series:"
    )

    print(
        series[
            "replay_count"
        ]
        .value_counts()
        .sort_index()
    )

    incomplete = series[
        ~series[
            "winner_reaches_required_wins"
        ]
    ]

    orientation_issue_series = series[
        series[
            "orientation_error"
        ]
    ]

    inconsistent_names = series[
        ~series[
            "team_name_consistency"
        ]
    ]

    print(
        f"\nSeries with orientation errors: "
        f"{len(orientation_issue_series)}"
    )

    print(
        "Series where observed winner does not "
        f"reach expected threshold: "
        f"{len(incomplete)}"
    )

    print(
        f"Series with inconsistent team names: "
        f"{len(inconsistent_names)}"
    )

    if len(incomplete) > 0:
        print(
            "\nSERIES REQUIRING INSPECTION:"
        )

        inspect_cols = [
            "series_id",
            "stage",
            "subgroup",
            "team_a",
            "team_b",
            "team_a_players",
            "team_b_players",
            "replay_count",
            "team_a_game_wins_observed",
            "team_b_game_wins_observed",
            "wins_required",
        ]

        print(
            incomplete[
                inspect_cols
            ].to_string(
                index=False
            )
        )

    if len(inconsistent_names) > 0:
        print(
            "\nTEAM-NAME CONSISTENCY ISSUES:"
        )

        print(
            inconsistent_names[
                [
                    "series_id",
                    "team_a",
                    "team_b",
                    "team_a_players",
                    "team_b_players",
                ]
            ].to_string(
                index=False
            )
        )

    unusual_player_counts = games[
        (
            games[
                "blue_player_count"
            ] != 3
        )
        |
        (
            games[
                "orange_player_count"
            ] != 3
        )
    ]

    print(
        f"\nGames with non-3v3 player counts: "
        f"{len(unusual_player_counts)}"
    )

    if len(unusual_player_counts) > 0:
        print(
            unusual_player_counts[
                [
                    "replay_id",
                    "series_id",
                    "blue_player_count",
                    "orange_player_count",
                    "blue_players",
                    "orange_players",
                ]
            ].to_string(
                index=False
            )
        )

    print("\nSaved:")
    print(
        config[
            "games_output"
        ]
    )
    print(
        config[
            "series_output"
        ]
    )
    print(
        config[
            "matchups_output"
        ]
    )

    return games, series


# ============================================================
# MAIN
# ============================================================

print("=" * 72)
print("RECONSTRUCTING OPEN 2 AND OPEN 3 SERIES")
print("=" * 72)

all_results = {}

for open_name, config in OPEN_CONFIG.items():

    games, series = process_open(
        open_name,
        config,
    )

    all_results[
        open_name
    ] = {
        "games": games,
        "series": series,
    }


# ============================================================
# FINAL SUMMARY
# ============================================================

divider("FINAL SUMMARY")

total_games = 0
total_series = 0
total_incomplete = 0
total_orientation_errors = 0

for open_name in [
    "Open 2",
    "Open 3",
]:

    games = all_results[
        open_name
    ]["games"]

    series = all_results[
        open_name
    ]["series"]

    incomplete_count = int(
        (
            ~series[
                "winner_reaches_required_wins"
            ]
        ).sum()
    )

    orientation_count = int(
        series[
            "orientation_error"
        ].sum()
    )

    print(
        f"{open_name}: "
        f"{len(games)} games, "
        f"{len(series)} series, "
        f"{incomplete_count} incomplete-threshold series, "
        f"{orientation_count} orientation-error series"
    )

    total_games += len(
        games
    )

    total_series += len(
        series
    )

    total_incomplete += (
        incomplete_count
    )

    total_orientation_errors += (
        orientation_count
    )

print(
    f"\nCombined games: "
    f"{total_games}"
)

print(
    f"Combined reconstructed series: "
    f"{total_series}"
)

print(
    f"Combined incomplete-threshold series: "
    f"{total_incomplete}"
)

print(
    f"Combined orientation-error series: "
    f"{total_orientation_errors}"
)

if (
    total_games == 286
    and total_incomplete == 0
    and total_orientation_errors == 0
):
    print(
        "\nPASS: all 286 games were retained and every "
        "reconstructed series reaches its expected win threshold."
    )
else:
    print(
        "\nCHECK REQUIRED: review any series shown above before "
        "building ML observations."
    )

print("\n" + "=" * 72)
print("DONE")
print("=" * 72)
