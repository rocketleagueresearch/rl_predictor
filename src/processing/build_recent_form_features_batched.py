from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from src.processing.player_aliases import normalise_player_name


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ballchasing"
)

INPUT_COMBINED = (
    PROCESSED_DIR
    / "open1_open2_open3_combined_ml_observations.csv"
)

OPEN1_SERIES = (
    PROCESSED_DIR
    / "open1_series_final.csv"
)

OPEN2_SERIES = (
    PROCESSED_DIR
    / "open2_series_reconstructed.csv"
)

OPEN3_SERIES = (
    PROCESSED_DIR
    / "open3_series_reconstructed.csv"
)

OUTPUT_FEATURES = (
    PROCESSED_DIR
    / "open1_open2_open3_recent_form_features.csv"
)

OUTPUT_AUDIT = (
    PROCESSED_DIR
    / "recent_form_feature_audit.csv"
)


# ============================================================
# SETTINGS
# ============================================================

TEAM_ALIASES = {
    "GK ESPORTS": "GEEKAY ESPORTS",
}

RECENT_WINDOWS = (3, 5)


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


def normalise_team_name(value):
    text = clean_text(value)

    if not text:
        return ""

    upper = text.upper()

    return TEAM_ALIASES.get(
        upper,
        upper,
    )


def split_roster(value):
    text = clean_text(value)

    if not text:
        return []

    # Existing files use pipe-delimited roster strings.
    players = [
        item.strip()
        for item in text.split("|")
        if item.strip()
    ]

    return players


def normalise_roster_signature(value):
    players = split_roster(value)

    if not players:
        return ""

    normalised = [
        normalise_player_name(player)
        for player in players
    ]

    normalised = [
        clean_text(player)
        for player in normalised
        if clean_text(player)
    ]

    return "|".join(
        sorted(
            normalised,
            key=lambda x: x.lower(),
        )
    )


def load_series_file(path, tournament_name):
    if not path.exists():
        raise FileNotFoundError(
            f"Series file not found:\n{path}"
        )

    frame = pd.read_csv(path)

    required = [
        "series_id",
        "team_a_game_wins_observed",
        "team_b_game_wins_observed",
        "target_team_a_win",
    ]

    missing = [
        col
        for col in required
        if col not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: "
            + ", ".join(missing)
        )

    keep = required.copy()

    if "team_a" in frame.columns:
        keep.append("team_a")

    if "team_b" in frame.columns:
        keep.append("team_b")

    frame = frame[
        keep
    ].copy()

    frame[
        "tournament_name"
    ] = tournament_name

    frame = frame.rename(
        columns={
            "team_a_game_wins_observed":
                "std_team_a_game_wins_observed",
            "team_b_game_wins_observed":
                "std_team_b_game_wins_observed",
            "target_team_a_win":
                "std_target_team_a_win",
            "team_a":
                "std_team_a",
            "team_b":
                "std_team_b",
        }
    )

    return frame


def build_roster_lookup(df):
    """
    Build same-tournament roster -> team-name mappings.

    Only unique mappings are retained.
    Ambiguous rosters are deliberately excluded.

    This is identity normalisation only.
    It does not use outcomes or performance.
    """

    roster_to_teams = defaultdict(set)

    for _, row in df.iterrows():

        tournament = clean_text(
            row.get(
                "tournament_name",
                "",
            )
        )

        for side in ("a", "b"):

            team_col = f"team_{side}"
            roster_col = f"team_{side}_players"

            team_name = normalise_team_name(
                row.get(
                    team_col,
                    "",
                )
            )

            roster_sig = normalise_roster_signature(
                row.get(
                    roster_col,
                    "",
                )
            )

            if (
                tournament
                and roster_sig
                and team_name
            ):
                roster_to_teams[
                    (
                        tournament,
                        roster_sig,
                    )
                ].add(
                    team_name
                )

    unique_lookup = {}
    ambiguous_lookup = {}

    for key, teams in roster_to_teams.items():

        sorted_teams = sorted(
            teams
        )

        if len(sorted_teams) == 1:
            unique_lookup[
                key
            ] = sorted_teams[0]
        else:
            ambiguous_lookup[
                key
            ] = sorted_teams

    return (
        unique_lookup,
        ambiguous_lookup,
    )


def resolve_identity(
    row,
    side,
    roster_lookup,
):
    team_col = f"team_{side}"
    roster_col = f"team_{side}_players"

    stored_team = normalise_team_name(
        row.get(
            team_col,
            "",
        )
    )

    if stored_team:
        return (
            stored_team,
            "stored_team_name",
        )

    tournament = clean_text(
        row.get(
            "tournament_name",
            "",
        )
    )

    roster_sig = normalise_roster_signature(
        row.get(
            roster_col,
            "",
        )
    )

    lookup_name = roster_lookup.get(
        (
            tournament,
            roster_sig,
        )
    )

    if lookup_name:
        return (
            lookup_name,
            "same_tournament_roster_lookup",
        )

    # Safety fallback:
    # use roster identity only when no unique team name can be established.
    # This keeps histories separate without inventing a team name.
    if roster_sig:
        return (
            f"ROSTER::{roster_sig}",
            "roster_fallback",
        )

    return (
        "",
        "unresolved",
    )


def new_history():
    return {
        "series_results": [],
        "game_win_rates": [],
        "series_wins": 0,
        "series_losses": 0,
        "game_wins": 0,
        "game_losses": 0,
        "current_win_streak": 0,
    }


def history_features(history):
    series_played = (
        history["series_wins"]
        +
        history["series_losses"]
    )

    games_played = (
        history["game_wins"]
        +
        history["game_losses"]
    )

    if series_played > 0:
        series_win_rate = (
            history["series_wins"]
            /
            series_played
        )
    else:
        series_win_rate = np.nan

    if games_played > 0:
        game_win_rate = (
            history["game_wins"]
            /
            games_played
        )
    else:
        game_win_rate = np.nan

    out = {
        "prior_series_played":
            series_played,
        "prior_series_wins":
            history["series_wins"],
        "prior_series_losses":
            history["series_losses"],
        "prior_series_win_rate":
            series_win_rate,
        "prior_game_wins":
            history["game_wins"],
        "prior_game_losses":
            history["game_losses"],
        "prior_game_win_rate":
            game_win_rate,
        "prior_series_win_streak":
            history["current_win_streak"],
    }

    for window in RECENT_WINDOWS:

        recent_results = (
            history[
                "series_results"
            ][-window:]
        )

        recent_game_rates = (
            history[
                "game_win_rates"
            ][-window:]
        )

        played = len(
            recent_results
        )

        out[
            f"recent{window}_series_played"
        ] = played

        if played > 0:
            out[
                f"recent{window}_series_win_rate"
            ] = float(
                np.mean(
                    recent_results
                )
            )

            out[
                f"recent{window}_avg_game_win_rate"
            ] = float(
                np.mean(
                    recent_game_rates
                )
            )
        else:
            out[
                f"recent{window}_series_win_rate"
            ] = np.nan

            out[
                f"recent{window}_avg_game_win_rate"
            ] = np.nan

    return out


def apply_history_update(
    history,
    won_series,
    game_wins,
    game_losses,
):
    won_series = int(
        won_series
    )

    game_wins = int(
        game_wins
    )

    game_losses = int(
        game_losses
    )

    history[
        "series_results"
    ].append(
        won_series
    )

    total_games = (
        game_wins
        +
        game_losses
    )

    if total_games > 0:
        game_rate = (
            game_wins
            /
            total_games
        )
    else:
        game_rate = np.nan

    history[
        "game_win_rates"
    ].append(
        game_rate
    )

    if won_series == 1:
        history[
            "series_wins"
        ] += 1

        history[
            "current_win_streak"
        ] += 1
    else:
        history[
            "series_losses"
        ] += 1

        history[
            "current_win_streak"
        ] = 0

    history[
        "game_wins"
    ] += game_wins

    history[
        "game_losses"
    ] += game_losses


# ============================================================
# LOAD BASE DATA
# ============================================================

divider(
    "BUILDING BATCH-SAFE RECENT-FORM FEATURES"
)

if not INPUT_COMBINED.exists():
    raise FileNotFoundError(
        f"Combined dataset not found:\n{INPUT_COMBINED}"
    )

df = pd.read_csv(
    INPUT_COMBINED
)

print(
    f"Combined rows loaded: {len(df)}"
)

if len(df) != 99:
    raise ValueError(
        f"Expected 99 combined rows, found {len(df)}."
    )

required_base = [
    "series_id",
    "tournament_name",
    "tournament_order",
    "series_start_utc",
    "team_a",
    "team_b",
    "team_a_players",
    "team_b_players",
    "target_team_a_win",
]

missing_base = [
    col
    for col in required_base
    if col not in df.columns
]

if missing_base:
    raise ValueError(
        "Combined dataset is missing required columns: "
        + ", ".join(
            missing_base
        )
    )

df[
    "series_start_utc"
] = pd.to_datetime(
    df[
        "series_start_utc"
    ],
    utc=True,
    errors="coerce",
)

print(
    "Missing timestamps: "
    f"{df['series_start_utc'].isna().sum()}"
)

if df[
    "series_start_utc"
].isna().any():
    raise ValueError(
        "Missing timestamps prevent strict temporal feature construction."
    )


# ============================================================
# STANDARDISED SERIES RESULTS
# ============================================================

open1 = load_series_file(
    OPEN1_SERIES,
    "Open 1",
)

open2 = load_series_file(
    OPEN2_SERIES,
    "Open 2",
)

open3 = load_series_file(
    OPEN3_SERIES,
    "Open 3",
)

print(
    f"Open 1 series rows loaded: {len(open1)}"
)

print(
    f"Open 2 series rows loaded: {len(open2)}"
)

print(
    f"Open 3 series rows loaded: {len(open3)}"
)

series_results = pd.concat(
    [
        open1,
        open2,
        open3,
    ],
    ignore_index=True,
)

if len(
    series_results
) != 99:
    raise ValueError(
        "Expected 99 standardised series rows."
    )

# Remove any potentially inconsistent score columns from the combined file
# before merging the trusted standardised series-result fields.
drop_if_present = [
    "team_a_game_wins_observed",
    "team_b_game_wins_observed",
    "std_team_a_game_wins_observed",
    "std_team_b_game_wins_observed",
    "std_target_team_a_win",
]

df = df.drop(
    columns=[
        col
        for col in drop_if_present
        if col in df.columns
    ],
    errors="ignore",
)

df = df.merge(
    series_results[
        [
            "series_id",
            "tournament_name",
            "std_team_a_game_wins_observed",
            "std_team_b_game_wins_observed",
            "std_target_team_a_win",
        ]
    ],
    on=[
        "series_id",
        "tournament_name",
    ],
    how="left",
    validate="one_to_one",
)

missing_scores = (
    df[
        [
            "std_team_a_game_wins_observed",
            "std_team_b_game_wins_observed",
        ]
    ]
    .isna()
    .any(
        axis=1
    )
    .sum()
)

print(
    "Rows missing standardised observed series score: "
    f"{missing_scores}"
)

if missing_scores != 0:
    raise ValueError(
        "Some rows are missing the standardised observed series score."
    )

target_mismatch = (
    pd.to_numeric(
        df[
            "target_team_a_win"
        ],
        errors="coerce",
    )
    !=
    pd.to_numeric(
        df[
            "std_target_team_a_win"
        ],
        errors="coerce",
    )
).sum()

print(
    "Target mismatches against standardised series files: "
    f"{target_mismatch}"
)

if target_mismatch != 0:
    raise ValueError(
        "Combined target differs from standardised series target."
    )


# ============================================================
# SAME-TOURNAMENT ROSTER IDENTITY LOOKUP
# ============================================================

(
    roster_lookup,
    ambiguous_lookup,
) = build_roster_lookup(
    df
)

print(
    "Tournament-roster identities with unique team mapping: "
    f"{len(roster_lookup)}"
)

print(
    "Ambiguous tournament-roster mappings: "
    f"{len(ambiguous_lookup)}"
)

if ambiguous_lookup:
    print(
        "\nAmbiguous mappings are NOT used:"
    )

    for (
        tournament,
        roster,
    ), teams in sorted(
        ambiguous_lookup.items()
    ):
        print(
            f"  {tournament} | {roster} -> {teams}"
        )


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

df = df.sort_values(
    [
        "series_start_utc",
        "tournament_order",
        "series_id",
    ]
).reset_index(
    drop=True
)

df[
    "chronological_index"
] = np.arange(
    1,
    len(df) + 1,
)


# ============================================================
# STRICT TIMESTAMP-BATCH FEATURE CONSTRUCTION
# ============================================================

histories = defaultdict(
    new_history
)

feature_rows = []

same_timestamp_groups = 0
same_timestamp_rows = 0

resolved_from_roster = 0
roster_fallback_rows = 0
unresolved_rows = 0

# IMPORTANT:
# All rows at timestamp T are FEATURED first.
# Only after every row at T is featured are histories updated.
for timestamp, batch in df.groupby(
    "series_start_utc",
    sort=True,
):

    batch = batch.sort_values(
        [
            "tournament_order",
            "series_id",
        ]
    )

    if len(batch) > 1:
        same_timestamp_groups += 1
        same_timestamp_rows += len(batch)

    pending_updates = []

    for idx, row in batch.iterrows():

        team_a_identity, source_a = resolve_identity(
            row,
            "a",
            roster_lookup,
        )

        team_b_identity, source_b = resolve_identity(
            row,
            "b",
            roster_lookup,
        )

        if (
            source_a
            ==
            "same_tournament_roster_lookup"
        ):
            resolved_from_roster += 1

        if (
            source_b
            ==
            "same_tournament_roster_lookup"
        ):
            resolved_from_roster += 1

        if (
            source_a
            ==
            "roster_fallback"
            or source_b
            ==
            "roster_fallback"
        ):
            roster_fallback_rows += 1

        if (
            not team_a_identity
            or not team_b_identity
        ):
            unresolved_rows += 1

        hist_a = histories[
            team_a_identity
        ]

        hist_b = histories[
            team_b_identity
        ]

        feat_a = history_features(
            hist_a
        )

        feat_b = history_features(
            hist_b
        )

        output = {
            "row_index": idx,
            "team_a_recent_form_name":
                team_a_identity,
            "team_b_recent_form_name":
                team_b_identity,
            "team_a_recent_form_identity_source":
                source_a,
            "team_b_recent_form_identity_source":
                source_b,
        }

        for name, value in feat_a.items():
            output[
                f"team_a_{name}"
            ] = value

        for name, value in feat_b.items():
            output[
                f"team_b_{name}"
            ] = value

        # Symmetric Team A minus Team B differences.
        output[
            "diff_prior_series_played"
        ] = (
            feat_a[
                "prior_series_played"
            ]
            -
            feat_b[
                "prior_series_played"
            ]
        )

        output[
            "diff_prior_series_win_rate"
        ] = (
            feat_a[
                "prior_series_win_rate"
            ]
            -
            feat_b[
                "prior_series_win_rate"
            ]
        )

        output[
            "diff_prior_game_win_rate"
        ] = (
            feat_a[
                "prior_game_win_rate"
            ]
            -
            feat_b[
                "prior_game_win_rate"
            ]
        )

        output[
            "diff_prior_series_win_streak"
        ] = (
            feat_a[
                "prior_series_win_streak"
            ]
            -
            feat_b[
                "prior_series_win_streak"
            ]
        )

        for window in RECENT_WINDOWS:

            output[
                f"diff_recent{window}_series_win_rate"
            ] = (
                feat_a[
                    f"recent{window}_series_win_rate"
                ]
                -
                feat_b[
                    f"recent{window}_series_win_rate"
                ]
            )

            output[
                f"diff_recent{window}_avg_game_win_rate"
            ] = (
                feat_a[
                    f"recent{window}_avg_game_win_rate"
                ]
                -
                feat_b[
                    f"recent{window}_avg_game_win_rate"
                ]
            )

        feature_rows.append(
            output
        )

        target_a = int(
            row[
                "std_target_team_a_win"
            ]
        )

        observed_a = int(
            row[
                "std_team_a_game_wins_observed"
            ]
        )

        observed_b = int(
            row[
                "std_team_b_game_wins_observed"
            ]
        )

        # Store updates only. Do NOT apply yet.
        pending_updates.append(
            {
                "team_a_identity":
                    team_a_identity,
                "team_b_identity":
                    team_b_identity,
                "team_a_won":
                    target_a,
                "team_b_won":
                    1 - target_a,
                "team_a_game_wins":
                    observed_a,
                "team_a_game_losses":
                    observed_b,
                "team_b_game_wins":
                    observed_b,
                "team_b_game_losses":
                    observed_a,
            }
        )

    # Now that every series at timestamp T has been featured from
    # histories containing only information from times < T,
    # apply all outcomes from T.
    for update in pending_updates:

        apply_history_update(
            histories[
                update[
                    "team_a_identity"
                ]
            ],
            won_series=
                update[
                    "team_a_won"
                ],
            game_wins=
                update[
                    "team_a_game_wins"
                ],
            game_losses=
                update[
                    "team_a_game_losses"
                ],
        )

        apply_history_update(
            histories[
                update[
                    "team_b_identity"
                ]
            ],
            won_series=
                update[
                    "team_b_won"
                ],
            game_wins=
                update[
                    "team_b_game_wins"
                ],
            game_losses=
                update[
                    "team_b_game_losses"
                ],
        )


# ============================================================
# MERGE FEATURES BACK
# ============================================================

features = pd.DataFrame(
    feature_rows
).set_index(
    "row_index"
)

for col in features.columns:
    df.loc[
        features.index,
        col,
    ] = features[
        col
    ]

# Restore strict chronological order.
df = df.sort_values(
    "chronological_index"
).reset_index(
    drop=True
)


# ============================================================
# AUDITS
# ============================================================

divider(
    "RECENT-FORM FEATURE AUDIT"
)

recent_feature_cols = [
    col
    for col in df.columns
    if (
        col.startswith(
            "team_a_prior_"
        )
        or col.startswith(
            "team_b_prior_"
        )
        or col.startswith(
            "team_a_recent"
        )
        or col.startswith(
            "team_b_recent"
        )
        or col.startswith(
            "diff_prior_"
        )
        or col.startswith(
            "diff_recent"
        )
    )
]

# Include identity metadata in the count, matching the original
# recent-form build convention.
identity_cols = [
    "team_a_recent_form_name",
    "team_b_recent_form_name",
    "team_a_recent_form_identity_source",
    "team_b_recent_form_identity_source",
]

created_cols = sorted(
    set(
        recent_feature_cols
        +
        identity_cols
    )
)

print(
    "Recent-form feature/identity columns created: "
    f"{len(created_cols)}"
)

print(
    f"Final dataset rows: {len(df)}"
)

print(
    f"Final dataset columns: {len(df.columns)}"
)

missing_identity = (
    (
        df[
            "team_a_recent_form_name"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        ==
        ""
    )
    |
    (
        df[
            "team_b_recent_form_name"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        ==
        ""
    )
).sum()

print(
    "Rows missing recent-form team identity: "
    f"{missing_identity}"
)

print(
    "Identity sides resolved from same-tournament roster mapping: "
    f"{resolved_from_roster}"
)

print(
    "Rows requiring unresolved roster fallback: "
    f"{roster_fallback_rows}"
)

print(
    "Rows with completely unresolved identity: "
    f"{unresolved_rows}"
)

print(
    "Same-timestamp groups processed as batches: "
    f"{same_timestamp_groups}"
)

print(
    "Series rows inside same-timestamp batches: "
    f"{same_timestamp_rows}"
)


# ============================================================
# FIRST-APPEARANCE LEAKAGE AUDIT
# ============================================================

seen_teams = set()
first_appearance_violations = []

for _, row in df.sort_values(
    [
        "series_start_utc",
        "tournament_order",
        "series_id",
    ]
).iterrows():

    team_a = row[
        "team_a_recent_form_name"
    ]

    team_b = row[
        "team_b_recent_form_name"
    ]

    prior_a = row[
        "team_a_prior_series_played"
    ]

    prior_b = row[
        "team_b_prior_series_played"
    ]

    if (
        team_a not in seen_teams
        and prior_a != 0
    ):
        first_appearance_violations.append(
            (
                row[
                    "series_id"
                ],
                "A",
                team_a,
                prior_a,
            )
        )

    if (
        team_b not in seen_teams
        and prior_b != 0
    ):
        first_appearance_violations.append(
            (
                row[
                    "series_id"
                ],
                "B",
                team_b,
                prior_b,
            )
        )

    seen_teams.add(
        team_a
    )

    seen_teams.add(
        team_b
    )

print(
    "First-appearance prior-count violations: "
    f"{len(first_appearance_violations)}"
)


# ============================================================
# STRICT SAME-TIMESTAMP AUDIT
# ============================================================

# For every shared timestamp, verify that repeated teams have the
# same pre-timestamp prior-series count across all simultaneous series.
# This directly checks the NIP-type case found by the audit.

same_time_consistency_violations = []

for timestamp, batch in df.groupby(
    "series_start_utc"
):

    if len(batch) <= 1:
        continue

    team_records = defaultdict(
        list
    )

    for _, row in batch.iterrows():

        team_records[
            row[
                "team_a_recent_form_name"
            ]
        ].append(
            row[
                "team_a_prior_series_played"
            ]
        )

        team_records[
            row[
                "team_b_recent_form_name"
            ]
        ].append(
            row[
                "team_b_prior_series_played"
            ]
        )

    for team_name, values in team_records.items():

        if len(values) > 1:

            unique_values = set(
                values
            )

            if len(unique_values) > 1:
                same_time_consistency_violations.append(
                    {
                        "series_start_utc":
                            timestamp,
                        "team":
                            team_name,
                        "prior_counts":
                            sorted(
                                unique_values
                            ),
                    }
                )

print(
    "Same-timestamp prior-history consistency violations: "
    f"{len(same_time_consistency_violations)}"
)


# ============================================================
# COVERAGE SUMMARY
# ============================================================

audit_rows = []

for tournament, group in df.groupby(
    "tournament_name",
    sort=False,
):

    audit_rows.append(
        {
            "tournament_name":
                tournament,
            "rows":
                len(group),
            "team_a_zero_history":
                int(
                    (
                        group[
                            "team_a_prior_series_played"
                        ]
                        ==
                        0
                    ).sum()
                ),
            "team_b_zero_history":
                int(
                    (
                        group[
                            "team_b_prior_series_played"
                        ]
                        ==
                        0
                    ).sum()
                ),
            "mean_team_a_prior_series":
                float(
                    group[
                        "team_a_prior_series_played"
                    ].mean()
                ),
            "mean_team_b_prior_series":
                float(
                    group[
                        "team_b_prior_series_played"
                    ].mean()
                ),
        }
    )

audit_df = pd.DataFrame(
    audit_rows
)

print()
print(
    audit_df.to_string(
        index=False
    )
)


# ============================================================
# FINAL VALIDATION
# ============================================================

if len(df) != 99:
    raise ValueError(
        "Final dataset does not contain 99 rows."
    )

if missing_scores != 0:
    raise ValueError(
        "Missing standardised series scores remain."
    )

if missing_identity != 0:
    raise ValueError(
        "Missing recent-form identities remain."
    )

if unresolved_rows != 0:
    raise ValueError(
        "Some team identities are completely unresolved."
    )

if first_appearance_violations:
    raise ValueError(
        "First-appearance leakage audit failed."
    )

if same_time_consistency_violations:
    raise ValueError(
        "Same-timestamp strict-history audit failed."
    )


# ============================================================
# SAVE
# ============================================================

# Remove temporary standardised target; retain observed score fields
# under the names expected by downstream feature construction.

df[
    "team_a_game_wins_observed"
] = df[
    "std_team_a_game_wins_observed"
]

df[
    "team_b_game_wins_observed"
] = df[
    "std_team_b_game_wins_observed"
]

df = df.drop(
    columns=[
        "std_team_a_game_wins_observed",
        "std_team_b_game_wins_observed",
        "std_target_team_a_win",
    ],
    errors="ignore",
)

df.to_csv(
    OUTPUT_FEATURES,
    index=False,
)

audit_df.to_csv(
    OUTPUT_AUDIT,
    index=False,
)

print()
print(
    "Saved batch-safe recent-form dataset:"
)

print(
    OUTPUT_FEATURES
)

print()
print(
    "Saved recent-form audit:"
)

print(
    OUTPUT_AUDIT
)

print()
print(
    "PASS: recent-form features were rebuilt using strict "
    "timestamp batches."
)

print(
    "Every series at time T was computed only from history "
    "with timestamp < T."
)

print()
print("=" * 72)
print("DONE")
print("=" * 72)
