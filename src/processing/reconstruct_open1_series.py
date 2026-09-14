
from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

GAMES_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "ballchasing"
    / "open1_games_corrected.csv"
)

METADATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "ballchasing"
    / "open1_replay_metadata_utc.csv"
)

RAW_OPEN1_DIR = (
    BASE_DIR
    / "data"
    / "raw"
    / "ballchasing"
    / "Open 1"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "ballchasing"
    / "open1_series_reconstructed.csv"
)

PLAYER_MATCHUP_OUTPUT = (
    BASE_DIR
    / "data"
    / "processed"
    / "ballchasing"
    / "open1_player_matchups.csv"
)


# ============================================================
# PLAYER NAME ALIASES
# ============================================================
#
# Ballchasing username -> known player identity
#
# This is NOT being used to identify teams.
# It is only used so that the same player is recognised
# consistently across replays.

PLAYER_ALIASES = {
    "bhavi": "AJG",
}


# ============================================================
# TEAM NAME ALIASES
# ============================================================

TEAM_ALIASES = {
    "VIT": "TEAM VITALITY",
    "VITALITY": "TEAM VITALITY",
    "NOVO ESPORTS": "NOVO",
    "GK": "GEEKAY ESPORTS",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalise_player_name(value):
    """Normalise a player name for matching."""

    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    # Apply known player alias
    value = PLAYER_ALIASES.get(value, value)

    return value


def normalise_team_name(value):
    """Normalise a team name without using it for matchup detection."""

    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    return TEAM_ALIASES.get(value, value)


def canonical_side(players):
    """
    Sort the players on one team so that player order does not matter.
    """

    players = [
        p for p in players
        if p is not None
    ]

    return tuple(sorted(players, key=str.lower))


def make_matchup_fingerprint(blue_players, orange_players):
    """
    Create an order-independent fingerprint for the two teams.

    Blue/orange colour does not matter.

    Example:

        Team 1:
        A, B, C

        Team 2:
        D, E, F

    produces the same fingerprint if the colours are reversed.
    """

    blue_side = canonical_side(blue_players)
    orange_side = canonical_side(orange_players)

    if not blue_side or not orange_side:
        return None

    # Sort the two sides so blue/orange reversal does not matter.
    sides = sorted(
        [
            "|".join(blue_side),
            "|".join(orange_side)
        ],
        key=str.lower
    )

    return " || ".join(sides)


def parse_player_csv(path):
    """
    Read one Ballchasing player CSV and return the six players,
    separated by blue/orange side.
    """

    try:
        df = pd.read_csv(
            path,
            sep=";",
            engine="python"
        )
    except Exception as exc:
        print(f"Could not read {path.name}: {exc}")
        return None

    required_columns = {
        "color",
        "player name"
    }

    missing = required_columns - set(df.columns)

    if missing:
        print(
            f"Missing columns in {path.name}: "
            f"{sorted(missing)}"
        )
        return None

    blue_players = []
    orange_players = []

    for _, row in df.iterrows():

        player = normalise_player_name(
            row["player name"]
        )

        if player is None:
            continue

        color = str(
            row["color"]
        ).strip().lower()

        if color == "blue":
            blue_players.append(player)

        elif color == "orange":
            orange_players.append(player)

    return {
        "blue_players": canonical_side(blue_players),
        "orange_players": canonical_side(orange_players),
    }


# ============================================================
# LOAD EXISTING GAME DATA
# ============================================================

print("Loading Open 1 replay data...")

games = pd.read_csv(GAMES_FILE)
metadata = pd.read_csv(METADATA_FILE)

games["replay_id"] = games["replay_id"].astype(str)
metadata["replay_id"] = metadata["replay_id"].astype(str)

metadata["replay_datetime_utc"] = pd.to_datetime(
    metadata["replay_datetime_utc"],
    utc=True,
    errors="coerce"
)

# Remove any previous timestamp column before merging.
games = games.drop(
    columns=["replay_datetime_utc"],
    errors="ignore"
)

games = games.merge(
    metadata[
        [
            "replay_id",
            "replay_datetime_utc",
            "matchguid"
        ]
    ],
    on="replay_id",
    how="left"
)

print()
print(f"Games loaded: {len(games)}")


# ============================================================
# FIND ALL RAW PLAYER FILES
# ============================================================

print()
print("Searching for raw player files...")

player_files = sorted(
    RAW_OPEN1_DIR.rglob("*-players.csv")
)

print(
    f"Player CSV files found: {len(player_files)}"
)


# ============================================================
# READ PLAYER FILES
# ============================================================

player_rows = []

for index, path in enumerate(player_files, start=1):

    replay_id = path.name.replace(
        "-players.csv",
        ""
    )

    parsed = parse_player_csv(path)

    if parsed is None:
        continue

    blue_players = parsed["blue_players"]
    orange_players = parsed["orange_players"]

    # We expect three players on each side for normal
    # Rocket League 3v3 games, but we do NOT reject games
    # based on player count.

    matchup_fingerprint = make_matchup_fingerprint(
        blue_players,
        orange_players
    )

    player_rows.append(
        {
            "replay_id": replay_id,
            "player_file": str(path),
            "blue_players": " | ".join(blue_players),
            "orange_players": " | ".join(orange_players),
            "blue_player_count": len(blue_players),
            "orange_player_count": len(orange_players),
            "matchup_fingerprint": matchup_fingerprint,
        }
    )


player_matchups = pd.DataFrame(player_rows)

print(
    f"Player files successfully processed: "
    f"{len(player_matchups)}"
)


# ============================================================
# MERGE PLAYER MATCHUPS INTO GAME DATA
# ============================================================

games = games.merge(
    player_matchups,
    on="replay_id",
    how="left"
)


# ============================================================
# TIMESTAMP VALIDATION
# ============================================================

print()
print("Timestamp validation")
print("--------------------")

timestamp_count = games[
    "replay_datetime_utc"
].notna().sum()

timestamp_missing = games[
    "replay_datetime_utc"
].isna().sum()

print(
    f"Games with timestamps: {timestamp_count}"
)

print(
    f"Games without timestamps: {timestamp_missing}"
)


# ============================================================
# MATCHUP VALIDATION
# ============================================================

print()
print("Player matchup validation")
print("-------------------------")

fingerprint_count = games[
    "matchup_fingerprint"
].notna().sum()

fingerprint_missing = games[
    "matchup_fingerprint"
].isna().sum()

print(
    f"Games with player matchup fingerprints: "
    f"{fingerprint_count}"
)

print(
    f"Games without player matchup fingerprints: "
    f"{fingerprint_missing}"
)


# ============================================================
# NORMALISE EXISTING TEAM NAMES
# ============================================================
#
# These are retained only as existing metadata.
# They are NOT used to construct the matchup fingerprint.

games["blue_team"] = games[
    "blue_team"
].apply(normalise_team_name)

games["orange_team"] = games[
    "orange_team"
].apply(normalise_team_name)


# ============================================================
# CREATE MATCH DATE
# ============================================================

games["match_date"] = (
    games["replay_datetime_utc"]
    .dt.date
)


# ============================================================
# CREATE PLAYER-BASED SERIES KEY
# ============================================================
#
# Primary identity:
#
#     date + six-player matchup
#
# The date is retained so that two separate tournament
# encounters involving the same roster pair do not get
# automatically merged into one series.

games["series_key"] = (
    games["match_date"].astype(str)
    + " | "
    + games[
        "matchup_fingerprint"
    ].fillna("UNKNOWN_PLAYER_MATCHUP")
)


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

games = games.sort_values(
    [
        "matchup_fingerprint",
        "replay_datetime_utc"
    ],
    na_position="last"
).reset_index(drop=True)


# ============================================================
# NUMBER GAMES WITHIN PLAYER-BASED SERIES
# ============================================================

games[
    "game_number_in_candidate_series"
] = (
    games.groupby(
        "series_key",
        dropna=False
    ).cumcount() + 1
)


# ============================================================
# CREATE READABLE TEAM PAIR
# ============================================================
#
# This is only for display/validation.
# It does NOT determine series membership.

def readable_team_pair(row):

    blue = row["blue_team"]
    orange = row["orange_team"]

    if pd.isna(blue) or pd.isna(orange):
        return "UNKNOWN TEAM NAMES"

    return " vs ".join(
        sorted(
            [blue, orange],
            key=str.lower
        )
    )


games["display_team_pair"] = games.apply(
    readable_team_pair,
    axis=1
)


# ============================================================
# SERIES SUMMARY
# ============================================================

summary_rows = []

for series_key, group in games.groupby(
    "series_key",
    dropna=False
):

    # Use the player fingerprint as the identity.
    fingerprint = group[
        "matchup_fingerprint"
    ].dropna()

    if len(fingerprint) > 0:
        fingerprint = fingerprint.iloc[0]
    else:
        fingerprint = None

    # Existing team names are only displayed.
    team_pairs = (
        group["display_team_pair"]
        .dropna()
        .unique()
    )

    if len(team_pairs) == 1:
        display_team_pair = team_pairs[0]
    elif len(team_pairs) > 1:
        display_team_pair = " / ".join(
            sorted(team_pairs)
        )
    else:
        display_team_pair = "UNKNOWN TEAM NAMES"

    # Game winners are based on the corrected goal-based
    # winner already created in open1_games_corrected.csv.

    winners = group[
        "game_winner"
    ].dropna().apply(
        normalise_team_name
    )

    winner_counts = winners.value_counts()

    if len(winner_counts) > 0:
        candidate_winner = winner_counts.index[0]
        candidate_wins = winner_counts.iloc[0]
    else:
        candidate_winner = None
        candidate_wins = 0

    summary_rows.append(
        {
            "series_key": series_key,
            "matchup_fingerprint": fingerprint,
            "display_team_pair": display_team_pair,
            "games_in_candidate_series": len(group),
            "candidate_winner": candidate_winner,
            "candidate_winner_game_wins": candidate_wins,
            "first_game_utc": group[
                "replay_datetime_utc"
            ].min(),
            "last_game_utc": group[
                "replay_datetime_utc"
            ].max(),
        }
    )


summary = pd.DataFrame(summary_rows)


# ============================================================
# SUMMARY OUTPUT
# ============================================================

print()
print("=" * 70)
print("PLAYER-BASED SERIES RECONSTRUCTION")
print("=" * 70)

print()
print(
    f"Total replay games: {len(games)}"
)

print(
    f"Player-based candidate series: "
    f"{len(summary)}"
)

print()
print("Candidate series by game count")
print("-------------------------------")

print(
    summary[
        "games_in_candidate_series"
    ]
    .value_counts()
    .sort_index()
    .rename_axis("games")
    .to_string()
)


# ============================================================
# DISPLAY SERIES
# ============================================================

print()
print("=" * 70)
print("CANDIDATE SERIES")
print("=" * 70)

print()

display_columns = [
    "series_key",
    "display_team_pair",
    "games_in_candidate_series",
    "candidate_winner",
    "first_game_utc",
    "last_game_utc",
]

print(
    summary[
        display_columns
    ].to_string(index=False)
)


# ============================================================
# DISPLAY GAMES WITH UNKNOWN TEAM NAMES
# ============================================================
#
# These games are NOT lost.
# Their player matchup is still available.

unknown_team_games = games[
    games["display_team_pair"]
    == "UNKNOWN TEAM NAMES"
]

print()
print("=" * 70)
print("GAMES WITH UNKNOWN TEAM NAME FIELDS")
print("=" * 70)

print()

if len(unknown_team_games) == 0:

    print("None.")

else:

    unknown_columns = [
        "replay_id",
        "replay_datetime_utc",
        "matchup_fingerprint",
        "blue_players",
        "orange_players",
        "game_winner",
    ]

    print(
        unknown_team_games[
            unknown_columns
        ].to_string(index=False)
    )


# ============================================================
# DISPLAY UNUSUAL SERIES
# ============================================================

unusual = summary[
    summary[
        "games_in_candidate_series"
    ].isin([1, 2, 6, 7])
]

print()
print("=" * 70)
print("UNUSUAL PLAYER-BASED SERIES")
print("=" * 70)

print()

if len(unusual) == 0:

    print("None.")

else:

    print(
        unusual[
            display_columns
        ].to_string(index=False)
    )


# ============================================================
# SAVE FULL GAME-LEVEL OUTPUT
# ============================================================

games.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SAVE PLAYER MATCHUP TABLE
# ============================================================

player_matchups.to_csv(
    PLAYER_MATCHUP_OUTPUT,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("OUTPUT")
print("=" * 70)

print()
print("Game-level reconstructed data:")
print(OUTPUT_FILE)

print()
print("Player matchup data:")
print(PLAYER_MATCHUP_OUTPUT)

print()
print("Finished.")
