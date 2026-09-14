from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ballchasing"
    / "open1_open2_open3_recent_form_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "recent_form_timestamp_leakage_audit.csv"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 72)
print("RECENT-FORM TIMESTAMP LEAKAGE AUDIT")
print("=" * 72)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

required = [
    "series_id",
    "tournament_name",
    "series_start_utc",
]

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing)
    )

df["series_start_utc"] = pd.to_datetime(
    df["series_start_utc"],
    utc=True,
    errors="coerce",
)

print(f"Rows loaded: {len(df)}")
print(
    "Missing timestamps: "
    f"{df['series_start_utc'].isna().sum()}"
)

if df["series_start_utc"].isna().any():
    raise ValueError(
        "Cannot perform strict timestamp audit because "
        "one or more timestamps are missing."
    )


# ============================================================
# EXACT DUPLICATE TIMESTAMP AUDIT
# ============================================================

counts = (
    df.groupby(
        "series_start_utc",
        dropna=False,
    )
    .size()
    .rename("series_at_timestamp")
    .reset_index()
)

duplicate_times = counts.loc[
    counts["series_at_timestamp"] > 1
].copy()

duplicate_rows = df.merge(
    duplicate_times,
    on="series_start_utc",
    how="inner",
)

duplicate_rows = duplicate_rows.sort_values(
    [
        "series_start_utc",
        "tournament_name",
        "series_id",
    ]
).reset_index(drop=True)

print()
print(
    "Unique exact timestamps: "
    f"{df['series_start_utc'].nunique()}"
)
print(
    "Exact timestamps shared by multiple series: "
    f"{len(duplicate_times)}"
)
print(
    "Series rows involved in shared timestamps: "
    f"{len(duplicate_rows)}"
)


# ============================================================
# TOURNAMENT-SCOPED AUDIT
# ============================================================

tournament_counts = (
    df.groupby(
        [
            "tournament_name",
            "series_start_utc",
        ],
        dropna=False,
    )
    .size()
    .rename("series_at_tournament_timestamp")
    .reset_index()
)

tournament_duplicates = tournament_counts.loc[
    tournament_counts[
        "series_at_tournament_timestamp"
    ]
    > 1
].copy()

print(
    "Same-tournament timestamps shared by multiple series: "
    f"{len(tournament_duplicates)}"
)


# ============================================================
# DISPLAY DUPLICATES IF PRESENT
# ============================================================

if len(duplicate_rows) > 0:

    print()
    print("=" * 72)
    print("SHARED TIMESTAMPS FOUND")
    print("=" * 72)

    display_cols = [
        col
        for col in [
            "series_start_utc",
            "series_at_timestamp",
            "series_id",
            "tournament_name",
            "stage",
            "subgroup",
            "team_a",
            "team_b",
            "team_a_recent_form_name",
            "team_b_recent_form_name",
        ]
        if col in duplicate_rows.columns
    ]

    print(
        duplicate_rows[
            display_cols
        ].to_string(index=False)
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    duplicate_rows.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "Audit rows saved to:"
    )
    print(OUTPUT_FILE)

    print()
    print(
        "ACTION REQUIRED: exact duplicate series timestamps exist."
    )
    print(
        "The recent-form builder should be changed to process "
        "same-timestamp series as a batch:"
    )
    print(
        "1. Compute features for every series at timestamp T "
        "using history strictly before T."
    )
    print(
        "2. Only after all features at T are computed, update "
        "team histories with all series at T."
    )
    print()
    print(
        "Do NOT freeze the current model results yet."
    )

else:

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        columns=[
            "series_start_utc",
            "series_id",
            "tournament_name",
        ]
    ).to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 72)
    print("PASS")
    print("=" * 72)
    print(
        "No two series share the same exact series_start_utc."
    )
    print(
        "Therefore the existing row-by-row recent-form update "
        "cannot allow a same-timestamp series to influence another."
    )
    print(
        "The recent-form ordering is strict with respect to the "
        "available replay timestamps."
    )
    print()
    print(
        "The three model results can proceed to final comparative "
        "evaluation without rebuilding recent-form features."
    )

print()
print("=" * 72)
print("DONE")
print("=" * 72)
