from pathlib import Path
import sys
import pandas as pd
import numpy as np


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.processing.player_aliases import normalise_player_name


PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ballchasing"
)

BLAST_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "blast"
    / "BLAST_Player_History.xlsx"
)

OPEN_CONFIG = {
    "Open 2": {
        "series_file": PROCESSED_DIR / "open2_series_reconstructed.csv",
        "output_file": PROCESSED_DIR / "open2_ml_observations.csv",
        "coverage_file": PROCESSED_DIR / "open2_blast_coverage_report.csv",
    },
    "Open 3": {
        "series_file": PROCESSED_DIR / "open3_series_reconstructed.csv",
        "output_file": PROCESSED_DIR / "open3_ml_observations.csv",
        "coverage_file": PROCESSED_DIR / "open3_blast_coverage_report.csv",
    },
}


# ============================================================
# BLAST HISTORICAL COLUMNS
# ============================================================

BLAST_STATS = [
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


def parse_roster(roster_text):
    """
    Roster strings are stored as:
    player1|player2|player3
    """

    text = clean_text(
        roster_text
    )

    if text == "":
        return []

    return [
        normalise_player_name(
            name
        )
        for name in text.split("|")
        if clean_text(name) != ""
    ]


def safe_numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    )


# ============================================================
# LOAD BLAST HISTORY
# ============================================================

def load_blast_history():

    if not BLAST_FILE.exists():
        raise FileNotFoundError(
            f"BLAST file not found:\n{BLAST_FILE}"
        )

    blast = pd.read_excel(
        BLAST_FILE,
        sheet_name="Sheet1",
    )

    if "player_name" not in blast.columns:
        raise ValueError(
            "BLAST sheet has no player_name column."
        )

    blast["player_name"] = (
        blast["player_name"]
        .astype(str)
        .str.strip()
    )

    blast["player_name_normalised"] = (
        blast["player_name"]
        .map(
            normalise_player_name
        )
    )

    for col in BLAST_STATS:

        if col not in blast.columns:
            blast[col] = np.nan

        blast[col] = safe_numeric(
            blast[col]
        )

    return blast


# ============================================================
# TEAM BLAST FEATURES
# ============================================================

def aggregate_team_history(
    roster_text,
    blast_lookup,
    team_prefix,
):

    players = parse_roster(
        roster_text
    )

    matched_names = []
    players_with_stats = []
    player_rows = []

    for player in players:

        row = blast_lookup.get(
            player
        )

        if row is None:
            continue

        matched_names.append(
            player
        )

        stat_values = [
            row.get(
                col,
                np.nan,
            )
            for col in BLAST_STATS
        ]

        if any(
            pd.notna(
                value
            )
            for value in stat_values
        ):
            players_with_stats.append(
                player
            )

        player_rows.append(
            row
        )

    output = {
        f"{team_prefix}_blast_players_total": len(players),
        f"{team_prefix}_blast_players_name_matched": len(matched_names),
        f"{team_prefix}_blast_players_with_stats": len(players_with_stats),
        f"{team_prefix}_blast_coverage_fraction": (
            len(players_with_stats)
            / len(players)
            if len(players) > 0
            else np.nan
        ),
    }

    for stat in BLAST_STATS:

        values = []

        for row in player_rows:

            value = row.get(
                stat,
                np.nan,
            )

            if pd.notna(
                value
            ):
                values.append(
                    float(value)
                )

        output[
            f"{team_prefix}_{stat}_mean"
        ] = (
            float(
                np.mean(values)
            )
            if len(values) > 0
            else np.nan
        )

    return output


# ============================================================
# PROCESS ONE OPEN
# ============================================================

def process_open(
    open_name,
    config,
    blast_lookup,
):

    divider(
        f"BUILDING {open_name.upper()} ML OBSERVATIONS"
    )

    series_file = config[
        "series_file"
    ]

    if not series_file.exists():
        raise FileNotFoundError(
            f"Series file not found:\n{series_file}"
        )

    series = pd.read_csv(
        series_file
    )

    print(
        f"\nSeries rows loaded: "
        f"{len(series)}"
    )

    required_cols = [
        "series_id",
        "open",
        "stage",
        "subgroup",
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
    ]

    missing_required = [
        col
        for col in required_cols
        if col not in series.columns
    ]

    if missing_required:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                missing_required
            )
        )

    series[
        "target_team_a_win"
    ] = pd.to_numeric(
        series[
            "target_team_a_win"
        ],
        errors="coerce",
    )

    missing_targets = int(
        series[
            "target_team_a_win"
        ].isna().sum()
    )

    invalid_targets = int(
        (
            ~series[
                "target_team_a_win"
            ].isin(
                [0, 1]
            )
            &
            series[
                "target_team_a_win"
            ].notna()
        ).sum()
    )

    print(
        f"Missing targets: "
        f"{missing_targets}"
    )

    print(
        f"Invalid targets: "
        f"{invalid_targets}"
    )

    records = []

    for _, row in series.iterrows():

        record = row.to_dict()

        team_a_features = (
            aggregate_team_history(
                row[
                    "team_a_players"
                ],
                blast_lookup,
                "team_a",
            )
        )

        team_b_features = (
            aggregate_team_history(
                row[
                    "team_b_players"
                ],
                blast_lookup,
                "team_b",
            )
        )

        record.update(
            team_a_features
        )

        record.update(
            team_b_features
        )

        for stat in BLAST_STATS:

            a_col = (
                f"team_a_{stat}_mean"
            )

            b_col = (
                f"team_b_{stat}_mean"
            )

            diff_col = (
                f"diff_{stat}"
            )

            a_val = record.get(
                a_col,
                np.nan,
            )

            b_val = record.get(
                b_col,
                np.nan,
            )

            if (
                pd.notna(
                    a_val
                )
                and
                pd.notna(
                    b_val
                )
            ):
                record[
                    diff_col
                ] = (
                    float(
                        a_val
                    )
                    -
                    float(
                        b_val
                    )
                )

            else:
                record[
                    diff_col
                ] = np.nan

        records.append(
            record
        )

    ml = pd.DataFrame(
        records
    )

    ml[
        "series_start_utc"
    ] = pd.to_datetime(
        ml[
            "series_start_utc"
        ],
        errors="coerce",
        utc=True,
    )

    ml = ml.sort_values(
        [
            "series_start_utc",
            "series_id",
        ]
    ).reset_index(
        drop=True
    )

    coverage_cols = [
        "series_id",
        "open",
        "stage",
        "team_a",
        "team_b",
        "team_a_players",
        "team_b_players",
        "team_a_blast_players_total",
        "team_a_blast_players_name_matched",
        "team_a_blast_players_with_stats",
        "team_a_blast_coverage_fraction",
        "team_b_blast_players_total",
        "team_b_blast_players_name_matched",
        "team_b_blast_players_with_stats",
        "team_b_blast_coverage_fraction",
    ]

    coverage = ml[
        coverage_cols
    ].copy()

    ml.to_csv(
        config[
            "output_file"
        ],
        index=False,
    )

    coverage.to_csv(
        config[
            "coverage_file"
        ],
        index=False,
    )

    divider(
        f"{open_name.upper()} ML SUMMARY"
    )

    print(
        f"ML observations created: "
        f"{len(ml)}"
    )

    print(
        f"ML columns: "
        f"{len(ml.columns)}"
    )

    print(
        "\nTarget distribution:"
    )

    print(
        ml[
            "target_team_a_win"
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    print(
        "\nTeam A BLAST players with stats:"
    )

    print(
        ml[
            "team_a_blast_players_with_stats"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nTeam B BLAST players with stats:"
    )

    print(
        ml[
            "team_b_blast_players_with_stats"
        ]
        .value_counts()
        .sort_index()
    )

    diff_cols = [
        f"diff_{stat}"
        for stat in BLAST_STATS
    ]

    all_diff_missing = (
        ml[
            diff_cols
        ]
        .isna()
        .all(
            axis=1
        )
    )

    print(
        "\nSeries with all BLAST difference "
        f"features missing: "
        f"{int(all_diff_missing.sum())}"
    )

    unmatched_name_rows = (
        (
            ml[
                "team_a_blast_players_name_matched"
            ]
            <
            ml[
                "team_a_blast_players_total"
            ]
        )
        |
        (
            ml[
                "team_b_blast_players_name_matched"
            ]
            <
            ml[
                "team_b_blast_players_total"
            ]
        )
    )

    print(
        "Series with at least one BLAST "
        f"name unmatched: "
        f"{int(unmatched_name_rows.sum())}"
    )

    if unmatched_name_rows.any():

        print(
            "\nBLAST NAME-MATCH ISSUES:"
        )

        print(
            ml.loc[
                unmatched_name_rows,
                [
                    "series_id",
                    "team_a",
                    "team_b",
                    "team_a_players",
                    "team_b_players",
                    "team_a_blast_players_total",
                    "team_a_blast_players_name_matched",
                    "team_b_blast_players_total",
                    "team_b_blast_players_name_matched",
                ],
            ].to_string(
                index=False
            )
        )

    print("\nSaved:")
    print(
        config[
            "output_file"
        ]
    )
    print(
        config[
            "coverage_file"
        ]
    )

    return ml, coverage


# ============================================================
# MAIN
# ============================================================

print("=" * 72)
print("BUILDING OPEN 2 AND OPEN 3 ML OBSERVATIONS")
print("=" * 72)

blast = load_blast_history()

print(
    f"\nBLAST player rows loaded: "
    f"{len(blast)}"
)

blast_lookup = {}

for _, row in blast.iterrows():

    key = clean_text(
        row[
            "player_name_normalised"
        ]
    )

    if key == "":
        continue

    blast_lookup[
        key
    ] = row.to_dict()

results = {}

for open_name, config in OPEN_CONFIG.items():

    ml, coverage = process_open(
        open_name,
        config,
        blast_lookup,
    )

    results[
        open_name
    ] = {
        "ml": ml,
        "coverage": coverage,
    }


# ============================================================
# FINAL SUMMARY
# ============================================================

divider(
    "FINAL SUMMARY"
)

total_rows = 0
total_missing_targets = 0
total_unmatched_name_series = 0

for open_name in [
    "Open 2",
    "Open 3",
]:

    ml = results[
        open_name
    ]["ml"]

    missing_targets = int(
        ml[
            "target_team_a_win"
        ].isna().sum()
    )

    unmatched_name_series = int(
        (
            (
                ml[
                    "team_a_blast_players_name_matched"
                ]
                <
                ml[
                    "team_a_blast_players_total"
                ]
            )
            |
            (
                ml[
                    "team_b_blast_players_name_matched"
                ]
                <
                ml[
                    "team_b_blast_players_total"
                ]
            )
        ).sum()
    )

    print(
        f"{open_name}: "
        f"{len(ml)} rows, "
        f"{missing_targets} missing targets, "
        f"{unmatched_name_series} series with BLAST name mismatches"
    )

    total_rows += len(
        ml
    )

    total_missing_targets += (
        missing_targets
    )

    total_unmatched_name_series += (
        unmatched_name_series
    )

print(
    f"\nCombined ML observations: "
    f"{total_rows}"
)

print(
    f"Combined missing targets: "
    f"{total_missing_targets}"
)

print(
    "Combined series with BLAST name "
    f"mismatches: "
    f"{total_unmatched_name_series}"
)

if (
    total_rows == 66
    and
    total_missing_targets == 0
):
    print(
        "\nPASS: Open 2 and Open 3 ML observation "
        "datasets were created successfully."
    )

else:
    print(
        "\nCHECK REQUIRED: review the diagnostics "
        "above before combining datasets."
    )

print(
    "\n" + "=" * 72
)

print(
    "DONE"
)

print(
    "=" * 72
)
