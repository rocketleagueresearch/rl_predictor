import pandas as pd
from pathlib import Path


# ============================================================
# FILE PATHS
# ============================================================

GAMES_FILE = Path(
    "data/processed/ballchasing/open1_games_corrected.csv"
)

TIME_FILE = Path(
    "data/processed/ballchasing/open1_games_with_time_utc.csv"
)

GROUP_A_FILE = Path(
    "data/processed/ballchasing/open1_group_stage_schedule.csv"
)

GROUP_B_FILE = Path(
    "data/processed/ballchasing/open1_group_stage_schedule_b.csv"
)

OUTPUT_FILE = Path(
    "data/processed/ballchasing/open1_group_stage_reconciled_corrected.csv"
)


# ============================================================
# KNOWN REPLAY IDs
# ============================================================

# These three files are KC vs NIP replays that were incorrectly
# placed in the Group B folder.
CONTAMINANT_REPLAYS = {
    "1bf26963-2ce3-4b82-ab7d-d27c4c05e210",
    "393bd037-80d2-46b9-b86a-0460b55fc7da",
    "957ca0a0-68e4-493c-9b41-632fbfd5ec00",
}


# These three files are verified NIP vs Ascend games.
NIP_ASCEND_REPLAYS = {
    "5876a916-06cb-41ef-b415-106f46ee1f45",
    "ed2ffa0b-e2d1-4f42-ae6b-d078167bfa5a",
    "75ea63cd-ec20-41ac-ad08-7cdcad774ece",
}


# ============================================================
# TEAM NAME NORMALISATION
# ============================================================

ALIASES = {
    "SNG": "SYNERGY",
    "STG": "SAVE THE GAME",
    "GK": "GEEKAY ESPORTS",
    "GK ESPORTS": "GEEKAY ESPORTS",
    "M8": "M8 ALPINE",
    "KC": "KARMINE CORP",
    "VIT": "TEAM VITALITY",
    "DIG": "DIGNITAS",
    "CLNT": "CALIENTE",
    "ASC": "ASCEND",
    "RED": "REDEMPTION",
    "SEL": "SELECAO",
    "SP": "STARTPOINT",
    "100": "100%",
    "NOVO ESPORTS": "NOVO",
}


def normalize_team(name):

    if pd.isna(name):
        return None

    name = str(name).strip().upper()

    return ALIASES.get(name, name)


def make_team_pair(team_a, team_b):

    if team_a is None or team_b is None:
        return None

    return tuple(sorted([team_a, team_b]))


# ============================================================
# LOAD DATA
# ============================================================

print("Loading data...")
print()

games = pd.read_csv(GAMES_FILE)

times = pd.read_csv(TIME_FILE)

schedule_a = pd.read_csv(GROUP_A_FILE)

schedule_b = pd.read_csv(GROUP_B_FILE)

print("Games loaded:", len(games))
print("Timestamp rows:", len(times))
print("Group A series:", len(schedule_a))
print("Group B series:", len(schedule_b))


# ============================================================
# ADD UTC TIMESTAMPS
# ============================================================

# The timestamp file already contains replay_datetime_utc.
# We use that column directly.

times = times[
    ["replay_id", "replay_datetime_utc"]
].copy()

times["replay_datetime_utc"] = pd.to_datetime(
    times["replay_datetime_utc"],
    errors="coerce",
    utc=True
)


# Remove any old timestamp column before merging.

games = games.drop(
    columns=["replay_datetime_utc"],
    errors="ignore"
)

games = games.merge(
    times,
    on="replay_id",
    how="left"
)


# ============================================================
# REMOVE KNOWN CONTAMINANTS
# ============================================================

before_count = len(games)

games = games[
    ~games["replay_id"].isin(CONTAMINANT_REPLAYS)
].copy()

removed_count = before_count - len(games)

print()
print(
    "Known contaminant replays removed:",
    removed_count
)


# ============================================================
# NORMALISE TEAM NAMES
# ============================================================

games["blue_team_norm"] = games[
    "blue_team"
].apply(normalize_team)

games["orange_team_norm"] = games[
    "orange_team"
].apply(normalize_team)


# ============================================================
# RECOVER NIP VS ASCEND GAMES
# ============================================================

# The player inspection established that:
#
# blue   = ASCEND
# orange = NIP
#
# for all three verified blank-team replays.

for replay_id in NIP_ASCEND_REPLAYS:

    mask = games["replay_id"] == replay_id

    games.loc[
        mask,
        "blue_team_norm"
    ] = "ASCEND"

    games.loc[
        mask,
        "orange_team_norm"
    ] = "NIP"


# Recalculate the winner for these games directly from goals.

games.loc[
    games["replay_id"].isin(NIP_ASCEND_REPLAYS)
    & (
        games["blue_goals"]
        >
        games["orange_goals"]
    ),
    "game_winner"
] = "ASCEND"


games.loc[
    games["replay_id"].isin(NIP_ASCEND_REPLAYS)
    & (
        games["orange_goals"]
        >
        games["blue_goals"]
    ),
    "game_winner"
] = "NIP"


# ============================================================
# CREATE TEAM PAIRS
# ============================================================

games["team_pair"] = games.apply(
    lambda row: make_team_pair(
        row["blue_team_norm"],
        row["orange_team_norm"]
    ),
    axis=1
)


# ============================================================
# PREPARE OFFICIAL SCHEDULE
# ============================================================

schedule = pd.concat(
    [
        schedule_a,
        schedule_b
    ],
    ignore_index=True
)

schedule["team_a_norm"] = schedule[
    "team_a"
].apply(normalize_team)

schedule["team_b_norm"] = schedule[
    "team_b"
].apply(normalize_team)

schedule["team_pair"] = schedule.apply(
    lambda row: make_team_pair(
        row["team_a_norm"],
        row["team_b_norm"]
    ),
    axis=1
)


# ============================================================
# MATCH REPLAYS TO OFFICIAL SERIES
# ============================================================

results = []

for _, game in games.iterrows():

    replay_id = game["replay_id"]

    pair = game["team_pair"]

    # --------------------------------------------------------
    # No identifiable teams
    # --------------------------------------------------------

    if pair is None:

        results.append({
            "replay_id": replay_id,
            "series_id": None,
            "match_status": "NO_TEAM_PAIR"
        })

        continue


    # --------------------------------------------------------
    # Find official series with same two teams
    # --------------------------------------------------------

    possible = schedule[
        schedule["team_pair"] == pair
    ].copy()


    # --------------------------------------------------------
    # No official series found
    # --------------------------------------------------------

    if possible.empty:

        results.append({
            "replay_id": replay_id,
            "series_id": None,
            "match_status": "NO_SCHEDULE_MATCH"
        })

        continue


    # --------------------------------------------------------
    # Exactly one official series
    # --------------------------------------------------------

    if len(possible) == 1:

        series = possible.iloc[0]

        results.append({
            "replay_id": replay_id,
            "series_id": series["id"],
            "match_status": "MATCHED"
        })

        continue


    # --------------------------------------------------------
    # Duplicate official team pairing
    #
    # We deliberately do NOT guess.
    # --------------------------------------------------------

    results.append({
        "replay_id": replay_id,
        "series_id": None,
        "match_status": "AMBIGUOUS"
    })


# Convert matching results to DataFrame.

matches = pd.DataFrame(results)


# ============================================================
# MERGE SERIES ASSIGNMENTS
# ============================================================

games = games.merge(
    matches,
    on="replay_id",
    how="left"
)


# ============================================================
# ORDER GAMES WITHIN EACH SERIES
# ============================================================

games = games.sort_values(
    [
        "series_id",
        "replay_datetime_utc"
    ]
)

games["game_number"] = (
    games
    .groupby("series_id")
    .cumcount()
    + 1
)


# ============================================================
# SAVE RECONCILED DATA
# ============================================================

games.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("RECONCILIATION SUMMARY")
print("=" * 70)

print()

print(
    "Games after contaminant removal:",
    len(games)
)

print(
    "Games matched to official series:",
    games["series_id"].notna().sum()
)

print(
    "Games without official series:",
    games["series_id"].isna().sum()
)


# ============================================================
# SERIES STATUS
# ============================================================

print()
print("Series status")
print("-------------")

status_counts = {
    "COMPLETE": 0,
    "CHECK": 0,
    "MISSING": 0
}

for _, series in schedule.iterrows():

    series_id = series["id"]

    series_games = games[
        games["series_id"] == series_id
    ]

    actual_games = len(series_games)

    expected_games = int(
        series["expected_games"]
    )

    if actual_games == expected_games:

        status = "COMPLETE"

    elif actual_games == 0:

        status = "MISSING"

    else:

        status = "CHECK"

    status_counts[status] += 1

    print(
        f"{series_id:10s} "
        f"{series['team_a']} vs "
        f"{series['team_b']} "
        f"{actual_games}/{expected_games} "
        f"{status}"
    )


# ============================================================
# WINNER VALIDATION
# ============================================================

print()
print("Winner validation")
print("-----------------")

winner_status_counts = {
    "CORRECT": 0,
    "CHECK": 0,
    "NO_GAMES": 0
}

for _, series in schedule.iterrows():

    series_id = series["id"]

    series_games = games[
        games["series_id"] == series_id
    ]

    official_winner = normalize_team(
        series["winner"]
    )


    # --------------------------------------------------------
    # No replay games
    # --------------------------------------------------------

    if series_games.empty:

        winner_status_counts["NO_GAMES"] += 1

        print(
            f"{series_id:10s} "
            f"{official_winner} "
            f"0-0 "
            f"NO_GAMES"
        )

        continue


    # --------------------------------------------------------
    # Count individual game winners
    # --------------------------------------------------------

    winner_games = (
        series_games["game_winner"]
        .apply(normalize_team)
        == official_winner
    ).sum()

    loser_games = (
        series_games["game_winner"]
        .notna()
    ).sum() - winner_games


    # --------------------------------------------------------
    # Validate against official series result
    # --------------------------------------------------------

    expected_winner_games = int(
        series["winner_games"]
    )

    expected_loser_games = int(
        series["loser_games"]
    )


    if (
        winner_games == expected_winner_games
        and
        loser_games == expected_loser_games
    ):

        status = "CORRECT"

    else:

        status = "CHECK"


    winner_status_counts[status] += 1

    print(
        f"{series_id:10s} "
        f"{official_winner} "
        f"{winner_games}-{loser_games} "
        f"{status}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print()

print(
    "Series COMPLETE:",
    status_counts["COMPLETE"]
)

print(
    "Series CHECK:",
    status_counts["CHECK"]
)

print(
    "Series MISSING:",
    status_counts["MISSING"]
)

print()

print(
    "Winner validation CORRECT:",
    winner_status_counts["CORRECT"]
)

print(
    "Winner validation CHECK:",
    winner_status_counts["CHECK"]
)

print(
    "Winner validation NO_GAMES:",
    winner_status_counts["NO_GAMES"]
)

print()
print("Saved to:")
print(OUTPUT_FILE)

print()
print("Finished.")