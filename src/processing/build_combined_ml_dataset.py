from pathlib import Path
import pandas as pd
import numpy as np


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

OPEN_FILES = {
    "Open 1": PROCESSED_DIR / "open1_ml_observations.csv",
    "Open 2": PROCESSED_DIR / "open2_ml_observations.csv",
    "Open 3": PROCESSED_DIR / "open3_ml_observations.csv",
}

OUTPUT_FILE = (
    PROCESSED_DIR
    / "open1_open2_open3_combined_ml_observations.csv"
)

COLUMN_REPORT_FILE = (
    PROCESSED_DIR
    / "combined_ml_column_alignment_report.csv"
)


# ============================================================
# HELPERS
# ============================================================

def divider(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def load_open(open_name, path):
    if not path.exists():
        raise FileNotFoundError(
            f"{open_name} file not found:\n{path}"
        )

    df = pd.read_csv(path)

    print(
        f"{open_name}: "
        f"{len(df)} rows, "
        f"{len(df.columns)} columns"
    )

    return df


# ============================================================
# LOAD
# ============================================================

divider("COMBINING OPEN 1, OPEN 2, AND OPEN 3 ML OBSERVATIONS")

datasets = {}

for open_name, path in OPEN_FILES.items():
    datasets[open_name] = load_open(
        open_name,
        path,
    )


# ============================================================
# BASIC EXPECTATION CHECKS
# ============================================================

expected_rows = {
    "Open 1": 33,
    "Open 2": 33,
    "Open 3": 33,
}

for open_name, expected in expected_rows.items():

    actual = len(
        datasets[
            open_name
        ]
    )

    if actual != expected:
        print(
            f"WARNING: {open_name} has "
            f"{actual} rows; expected {expected}."
        )


# ============================================================
# COLUMN ALIGNMENT REPORT
# ============================================================

all_columns = sorted(
    set().union(
        *[
            set(df.columns)
            for df in datasets.values()
        ]
    )
)

report_rows = []

for col in all_columns:

    row = {
        "column": col,
    }

    for open_name, df in datasets.items():
        row[
            f"in_{open_name.lower().replace(' ', '_')}"
        ] = (
            col in df.columns
        )

    report_rows.append(
        row
    )

column_report = pd.DataFrame(
    report_rows
)

column_report.to_csv(
    COLUMN_REPORT_FILE,
    index=False,
)

common_columns = set(
    datasets["Open 1"].columns
)

for open_name in [
    "Open 2",
    "Open 3",
]:
    common_columns &= set(
        datasets[
            open_name
        ].columns
    )

print(
    f"\nColumns appearing in all three Opens: "
    f"{len(common_columns)}"
)

print(
    f"Union of all columns: "
    f"{len(all_columns)}"
)

non_common = column_report[
    ~(
        column_report[
            [
                "in_open_1",
                "in_open_2",
                "in_open_3",
            ]
        ].all(
            axis=1
        )
    )
]

print(
    f"Columns not shared by all three Opens: "
    f"{len(non_common)}"
)

if len(non_common) > 0:
    print(
        "\nColumns not present in every Open:"
    )

    print(
        non_common.to_string(
            index=False
        )
    )


# ============================================================
# ADD TOURNAMENT ORDER
# ============================================================

open_order = {
    "Open 1": 1,
    "Open 2": 2,
    "Open 3": 3,
}

prepared = []

for open_name, df in datasets.items():

    temp = df.copy()

    temp["tournament_name"] = open_name

    temp["tournament_order"] = (
        open_order[
            open_name
        ]
    )

    # Prefer the explicit tournament name we control.
    # Keep any existing "open" column unchanged.
    prepared.append(
        temp
    )


# ============================================================
# COMBINE WITH COLUMN UNION
# ============================================================

combined = pd.concat(
    prepared,
    ignore_index=True,
    sort=False,
)


# ============================================================
# TEMPORAL ORDER
# ============================================================

if "series_start_utc" not in combined.columns:
    raise ValueError(
        "Combined data has no series_start_utc column."
    )

combined[
    "series_start_utc"
] = pd.to_datetime(
    combined[
        "series_start_utc"
    ],
    errors="coerce",
    utc=True,
)

missing_dates = int(
    combined[
        "series_start_utc"
    ].isna().sum()
)

print(
    f"\nRows with missing series_start_utc: "
    f"{missing_dates}"
)

combined = combined.sort_values(
    [
        "series_start_utc",
        "tournament_order",
        "series_id",
    ]
).reset_index(
    drop=True
)

combined[
    "chronological_index"
] = np.arange(
    1,
    len(combined) + 1,
)


# ============================================================
# TARGET CHECKS
# ============================================================

if "target_team_a_win" not in combined.columns:
    raise ValueError(
        "Combined data has no target_team_a_win column."
    )

combined[
    "target_team_a_win"
] = pd.to_numeric(
    combined[
        "target_team_a_win"
    ],
    errors="coerce",
)

missing_targets = int(
    combined[
        "target_team_a_win"
    ].isna().sum()
)

invalid_targets = int(
    (
        ~combined[
            "target_team_a_win"
        ].isin(
            [0, 1]
        )
        &
        combined[
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


# ============================================================
# DUPLICATE CHECKS
# ============================================================

duplicate_full_rows = int(
    combined.duplicated().sum()
)

duplicate_series_ids = int(
    combined[
        "series_id"
    ].duplicated().sum()
)

print(
    f"Duplicate full rows: "
    f"{duplicate_full_rows}"
)

print(
    f"Duplicate series IDs: "
    f"{duplicate_series_ids}"
)


# ============================================================
# CHRONOLOGY CHECKS
# ============================================================

divider("CHRONOLOGICAL COVERAGE")

for open_name in [
    "Open 1",
    "Open 2",
    "Open 3",
]:

    subset = combined[
        combined[
            "tournament_name"
        ]
        ==
        open_name
    ]

    print(
        f"{open_name}: "
        f"{len(subset)} rows"
    )

    if len(subset) > 0:

        print(
            f"  First series time: "
            f"{subset['series_start_utc'].min()}"
        )

        print(
            f"  Last series time:  "
            f"{subset['series_start_utc'].max()}"
        )


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

divider("TARGET DISTRIBUTION")

print(
    combined[
        "target_team_a_win"
    ]
    .value_counts(
        dropna=False
    )
    .sort_index()
)

print(
    "\nBy tournament:"
)

print(
    pd.crosstab(
        combined[
            "tournament_name"
        ],
        combined[
            "target_team_a_win"
        ],
        dropna=False,
    )
)


# ============================================================
# BLAST NAME-MATCH CHECK
# ============================================================

coverage_cols = [
    "team_a_blast_players_total",
    "team_a_blast_players_name_matched",
    "team_b_blast_players_total",
    "team_b_blast_players_name_matched",
]

if all(
    col in combined.columns
    for col in coverage_cols
):

    mismatch_mask = (
        (
            combined[
                "team_a_blast_players_name_matched"
            ]
            <
            combined[
                "team_a_blast_players_total"
            ]
        )
        |
        (
            combined[
                "team_b_blast_players_name_matched"
            ]
            <
            combined[
                "team_b_blast_players_total"
            ]
        )
    )

    print(
        "\nSeries with at least one BLAST "
        f"name mismatch: "
        f"{int(mismatch_mask.sum())}"
    )

else:
    print(
        "\nBLAST name-match columns are not "
        "shared by all rows; check column report."
    )


# ============================================================
# LEAKAGE-SENSITIVE COLUMNS REPORT
# ============================================================

divider("LEAKAGE-SENSITIVE COLUMNS")

keywords = [
    "winner",
    "target",
    "completed",
    "game_wins_observed",
    "replay_count",
]

leakage_candidates = []

for col in combined.columns:

    col_lower = col.lower()

    if any(
        keyword in col_lower
        for keyword in keywords
    ):
        leakage_candidates.append(
            col
        )

for col in leakage_candidates:
    print(col)

print(
    "\nThese columns are retained for dataset integrity, "
    "auditing, and target construction only. "
    "Do NOT automatically use them as model predictors."
)


# ============================================================
# SAVE
# ============================================================

combined.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

divider("FINAL SUMMARY")

print(
    f"Combined rows: "
    f"{len(combined)}"
)

print(
    f"Combined columns: "
    f"{len(combined.columns)}"
)

print(
    f"Missing timestamps: "
    f"{missing_dates}"
)

print(
    f"Missing targets: "
    f"{missing_targets}"
)

print(
    f"Invalid targets: "
    f"{invalid_targets}"
)

print(
    f"Duplicate series IDs: "
    f"{duplicate_series_ids}"
)

print(
    "\nSaved combined dataset:"
)

print(
    OUTPUT_FILE
)

print(
    "\nSaved column alignment report:"
)

print(
    COLUMN_REPORT_FILE
)

if (
    len(combined) == 99
    and
    missing_dates == 0
    and
    missing_targets == 0
    and
    invalid_targets == 0
    and
    duplicate_series_ids == 0
):

    print(
        "\nPASS: combined chronological dataset "
        "contains 99 valid series."
    )

else:

    print(
        "\nCHECK REQUIRED: review the diagnostics "
        "above before recent-form feature engineering."
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
