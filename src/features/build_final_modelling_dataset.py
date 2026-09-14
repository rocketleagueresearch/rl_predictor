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

INPUT_FILE = (
    PROCESSED_DIR
    / "open1_open2_open3_recent_form_features.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "final_modelling_dataset.csv"
)

FEATURE_LIST_FILE = (
    PROCESSED_DIR
    / "final_modelling_feature_list.csv"
)

MISSINGNESS_FILE = (
    PROCESSED_DIR
    / "final_modelling_missingness_report.csv"
)

FOLD_FILE = (
    PROCESSED_DIR
    / "temporal_fold_assignments.csv"
)


# ============================================================
# HELPERS
# ============================================================

def divider(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def first_existing(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def numeric_or_nan(df, col):
    if col not in df.columns:
        return pd.Series(
            np.nan,
            index=df.index,
            dtype="float64",
        )

    return pd.to_numeric(
        df[col],
        errors="coerce",
    )


# ============================================================
# LOAD
# ============================================================

divider("BUILDING FINAL MODELLING DATASET")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(
    f"Rows loaded: {len(df)}"
)

print(
    f"Columns loaded: {len(df.columns)}"
)

if len(df) != 99:
    print(
        f"WARNING: expected 99 rows, found {len(df)}."
    )


# ============================================================
# CORE VALIDATION
# ============================================================

required = [
    "series_id",
    "tournament_name",
    "tournament_order",
    "series_start_utc",
    "target_team_a_win",
]

missing_required = [
    col
    for col in required
    if col not in df.columns
]

if missing_required:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing_required)
    )

df[
    "series_start_utc"
] = pd.to_datetime(
    df[
        "series_start_utc"
    ],
    errors="coerce",
    utc=True,
)

df[
    "target_team_a_win"
] = pd.to_numeric(
    df[
        "target_team_a_win"
    ],
    errors="coerce",
)

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

missing_times = int(
    df[
        "series_start_utc"
    ].isna().sum()
)

missing_targets = int(
    df[
        "target_team_a_win"
    ].isna().sum()
)

invalid_targets = int(
    (
        df[
            "target_team_a_win"
        ].notna()
        &
        ~df[
            "target_team_a_win"
        ].isin(
            [0, 1]
        )
    ).sum()
)

duplicate_ids = int(
    df[
        "series_id"
    ].duplicated().sum()
)

print(
    f"Missing timestamps: {missing_times}"
)

print(
    f"Missing targets: {missing_targets}"
)

print(
    f"Invalid targets: {invalid_targets}"
)

print(
    f"Duplicate series IDs: {duplicate_ids}"
)


# ============================================================
# STANDARDISE BLAST COVERAGE
# ============================================================

# Open 1 used *_blast_stats_coverage.
# Open 2/3 used *_blast_coverage_fraction.
# Create one canonical feature for each side.

for side in [
    "a",
    "b",
]:

    fraction_col = (
        f"team_{side}_blast_coverage_fraction"
    )

    old_col = (
        f"team_{side}_blast_stats_coverage"
    )

    canonical_col = (
        f"team_{side}_blast_coverage"
    )

    fraction = numeric_or_nan(
        df,
        fraction_col,
    )

    old = numeric_or_nan(
        df,
        old_col,
    )

    canonical = fraction.copy()

    canonical = canonical.where(
        canonical.notna(),
        old,
    )

    df[
        canonical_col
    ] = canonical


# ============================================================
# FEATURE GROUPS
# ============================================================

# ------------------------------------------------------------
# BLAST historical difference features
# ------------------------------------------------------------

blast_diff_features = sorted(
    [
        col
        for col in df.columns
        if col.startswith(
            "diff_"
        )
        and (
            col.startswith(
                "diff_career_"
            )
            or
            col.startswith(
                "diff_2024_"
            )
        )
    ]
)


# ------------------------------------------------------------
# BLAST coverage / missingness diagnostics
# ------------------------------------------------------------

blast_coverage_features = [
    "team_a_blast_players_with_stats",
    "team_b_blast_players_with_stats",
    "team_a_blast_coverage",
    "team_b_blast_coverage",
]

blast_coverage_features = [
    col
    for col in blast_coverage_features
    if col in df.columns
]


# ------------------------------------------------------------
# Recent-form difference features
# ------------------------------------------------------------

recent_form_diff_features = [
    "diff_prior_series_played",
    "diff_prior_series_win_rate",
    "diff_prior_game_win_rate",
    "diff_prior_series_win_streak",
    "diff_recent3_series_win_rate",
    "diff_recent5_series_win_rate",
    "diff_recent3_avg_game_win_rate",
    "diff_recent5_avg_game_win_rate",
]

recent_form_diff_features = [
    col
    for col in recent_form_diff_features
    if col in df.columns
]


# ------------------------------------------------------------
# Recent-form coverage / experience features
# ------------------------------------------------------------

recent_form_coverage_features = [
    "team_a_prior_series_played",
    "team_b_prior_series_played",
    "team_a_recent3_series_played",
    "team_b_recent3_series_played",
    "team_a_recent5_series_played",
    "team_b_recent5_series_played",
]

recent_form_coverage_features = [
    col
    for col in recent_form_coverage_features
    if col in df.columns
]


# ------------------------------------------------------------
# Stage feature
# ------------------------------------------------------------

# Stage is known before the match and is therefore eligible.
# Encode as one simple binary feature:
# 1 = playoff, 0 = group stage.
#
# This is deterministic and does not use the result.

if "stage" in df.columns:

    stage_text = (
        df[
            "stage"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df[
        "is_playoff"
    ] = stage_text.str.contains(
        "playoff",
        regex=False,
    ).astype(int)

    stage_features = [
        "is_playoff",
    ]

else:
    stage_features = []


# ============================================================
# FINAL FEATURE WHITELIST
# ============================================================

feature_cols = (
    blast_diff_features
    +
    blast_coverage_features
    +
    recent_form_diff_features
    +
    recent_form_coverage_features
    +
    stage_features
)

# Preserve order while removing duplicates.
feature_cols = list(
    dict.fromkeys(
        feature_cols
    )
)

print(
    f"\nBLAST difference features: "
    f"{len(blast_diff_features)}"
)

print(
    f"BLAST coverage features: "
    f"{len(blast_coverage_features)}"
)

print(
    f"Recent-form difference features: "
    f"{len(recent_form_diff_features)}"
)

print(
    f"Recent-form coverage features: "
    f"{len(recent_form_coverage_features)}"
)

print(
    f"Stage features: "
    f"{len(stage_features)}"
)

print(
    f"Total predictor features: "
    f"{len(feature_cols)}"
)


# ============================================================
# FORCE PREDICTORS TO NUMERIC
# ============================================================

for col in feature_cols:
    df[
        col
    ] = pd.to_numeric(
        df[
            col
        ],
        errors="coerce",
    )


# ============================================================
# EXPLICIT LEAKAGE DEFENCE
# ============================================================

forbidden_exact = {
    "target_team_a_win",
    "winner_reaches_required_wins",
    "completed_team_a_game_wins",
    "completed_team_b_game_wins",
    "completed_series_winner_side",
    "team_a_game_wins_observed",
    "team_b_game_wins_observed",
    "series_winner_side",
    "series_winner",
    "replay_count",
    "game_winner",
    "result_reconciled",
    "result_source",
}

forbidden_keywords = [
    "winner",
    "game_wins_observed",
    "completed_",
    "replay_count",
]

forbidden_selected = []

for col in feature_cols:

    lower = col.lower()

    if (
        col in forbidden_exact
        or
        any(
            keyword in lower
            for keyword in forbidden_keywords
        )
    ):
        forbidden_selected.append(
            col
        )

if forbidden_selected:
    raise ValueError(
        "LEAKAGE RISK: forbidden predictor columns selected: "
        + ", ".join(
            forbidden_selected
        )
    )

print(
    "\nLeakage whitelist check: PASS"
)


# ============================================================
# TEMPORAL FOLDS
# ============================================================

# Fold design:
#
# Open 1:
#   training history only; no out-of-sample test prediction.
#
# Fold 1:
#   train = Open 1
#   test  = Open 2
#
# Fold 2:
#   train = Open 1 + Open 2
#   test  = Open 3
#
# The final comparison can pool the out-of-sample predictions
# from Open 2 and Open 3.

df[
    "fold1_role"
] = "unused"

df.loc[
    df[
        "tournament_name"
    ]
    ==
    "Open 1",
    "fold1_role",
] = "train"

df.loc[
    df[
        "tournament_name"
    ]
    ==
    "Open 2",
    "fold1_role",
] = "test"


df[
    "fold2_role"
] = "unused"

df.loc[
    df[
        "tournament_name"
    ].isin(
        [
            "Open 1",
            "Open 2",
        ]
    ),
    "fold2_role",
] = "train"

df.loc[
    df[
        "tournament_name"
    ]
    ==
    "Open 3",
    "fold2_role",
] = "test"


print(
    "\nFold 1 roles:"
)

print(
    df[
        "fold1_role"
    ].value_counts()
)

print(
    "\nFold 2 roles:"
)

print(
    df[
        "fold2_role"
    ].value_counts()
)


# ============================================================
# FEATURE MISSINGNESS
# ============================================================

missingness_rows = []

for col in feature_cols:

    missing_count = int(
        df[
            col
        ].isna().sum()
    )

    missing_pct = (
        100.0
        *
        missing_count
        /
        len(df)
    )

    unique_non_missing = int(
        df[
            col
        ]
        .dropna()
        .nunique()
    )

    missingness_rows.append(
        {
            "feature": col,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "unique_non_missing": unique_non_missing,
            "constant_non_missing": (
                unique_non_missing <= 1
            ),
        }
    )

missingness = pd.DataFrame(
    missingness_rows
)

constant_features = (
    missingness.loc[
        missingness[
            "constant_non_missing"
        ],
        "feature",
    ]
    .tolist()
)

print(
    f"\nConstant/non-informative selected features: "
    f"{len(constant_features)}"
)

if constant_features:
    for col in constant_features:
        print(
            f"  {col}"
        )


# ============================================================
# MISSINGNESS INDICATORS
# ============================================================

# Add explicit missingness indicators for predictors that
# genuinely contain missing values.
#
# The raw NaNs remain NaN here. Actual median imputation must
# be fitted only on each training fold later.

indicator_cols = []

for col in feature_cols:

    if df[
        col
    ].isna().any():

        indicator_col = (
            f"{col}__missing"
        )

        df[
            indicator_col
        ] = (
            df[
                col
            ]
            .isna()
            .astype(int)
        )

        indicator_cols.append(
            indicator_col
        )

final_feature_cols = (
    feature_cols
    +
    indicator_cols
)

print(
    f"\nMissingness indicators added: "
    f"{len(indicator_cols)}"
)

print(
    f"Final model-input feature count: "
    f"{len(final_feature_cols)}"
)


# ============================================================
# IMPORTANT: NO IMPUTATION HERE
# ============================================================

print(
    "\nNo missing values have been filled."
)

print(
    "Median imputation must be learned separately "
    "inside each temporal training fold."
)

print(
    "Scaling must also be learned from the training "
    "portion only for Logistic Regression and the Neural Network."
)


# ============================================================
# MODEL DATASET
# ============================================================

metadata_cols = [
    "chronological_index",
    "series_id",
    "tournament_name",
    "tournament_order",
    "series_start_utc",
    "stage",
    "subgroup",
    "team_a",
    "team_b",
    "team_a_players",
    "team_b_players",
    "team_a_recent_form_name",
    "team_b_recent_form_name",
    "team_a_recent_form_identity_source",
    "team_b_recent_form_identity_source",
    "fold1_role",
    "fold2_role",
]

metadata_cols = [
    col
    for col in metadata_cols
    if col in df.columns
]

model_cols = (
    metadata_cols
    +
    final_feature_cols
    +
    [
        "target_team_a_win",
    ]
)

model_df = df[
    model_cols
].copy()


# ============================================================
# FEATURE LIST REPORT
# ============================================================

feature_report_rows = []

for col in final_feature_cols:

    if col in blast_diff_features:
        group = "blast_historical_difference"

    elif col in blast_coverage_features:
        group = "blast_coverage"

    elif col in recent_form_diff_features:
        group = "recent_form_difference"

    elif col in recent_form_coverage_features:
        group = "recent_form_coverage"

    elif col in stage_features:
        group = "prematch_stage"

    elif col.endswith(
        "__missing"
    ):
        group = "missingness_indicator"

    else:
        group = "other"

    feature_report_rows.append(
        {
            "feature": col,
            "feature_group": group,
            "model_predictor": True,
        }
    )

feature_report = pd.DataFrame(
    feature_report_rows
)


# ============================================================
# FOLD ASSIGNMENT REPORT
# ============================================================

fold_report_cols = [
    "chronological_index",
    "series_id",
    "tournament_name",
    "series_start_utc",
    "fold1_role",
    "fold2_role",
    "target_team_a_win",
]

fold_report = df[
    fold_report_cols
].copy()


# ============================================================
# SAVE
# ============================================================

model_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

feature_report.to_csv(
    FEATURE_LIST_FILE,
    index=False,
)

missingness.to_csv(
    MISSINGNESS_FILE,
    index=False,
)

fold_report.to_csv(
    FOLD_FILE,
    index=False,
)


# ============================================================
# FINAL AUDIT
# ============================================================

divider("FINAL MODELLING DATASET AUDIT")

print(
    f"Rows: "
    f"{len(model_df)}"
)

print(
    f"Metadata columns: "
    f"{len(metadata_cols)}"
)

print(
    f"Base predictor features: "
    f"{len(feature_cols)}"
)

print(
    f"Missingness indicators: "
    f"{len(indicator_cols)}"
)

print(
    f"Total model-input features: "
    f"{len(final_feature_cols)}"
)

print(
    f"Targets missing: "
    f"{int(model_df['target_team_a_win'].isna().sum())}"
)

print(
    "\nTarget distribution:"
)

print(
    model_df[
        "target_team_a_win"
    ]
    .value_counts()
    .sort_index()
)

print(
    "\nFeature groups:"
)

print(
    feature_report[
        "feature_group"
    ].value_counts()
)

print(
    "\nHighest feature missingness:"
)

print(
    missingness.sort_values(
        [
            "missing_pct",
            "feature",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .head(
        20
    )
    .to_string(
        index=False
    )
)


# ============================================================
# PASS / FAIL
# ============================================================

fold1_train = int(
    (
        df[
            "fold1_role"
        ]
        ==
        "train"
    ).sum()
)

fold1_test = int(
    (
        df[
            "fold1_role"
        ]
        ==
        "test"
    ).sum()
)

fold2_train = int(
    (
        df[
            "fold2_role"
        ]
        ==
        "train"
    ).sum()
)

fold2_test = int(
    (
        df[
            "fold2_role"
        ]
        ==
        "test"
    ).sum()
)

print(
    "\nTemporal fold sizes:"
)

print(
    f"Fold 1: train={fold1_train}, "
    f"test={fold1_test}"
)

print(
    f"Fold 2: train={fold2_train}, "
    f"test={fold2_test}"
)

pass_checks = (
    len(model_df) == 99
    and
    missing_times == 0
    and
    missing_targets == 0
    and
    invalid_targets == 0
    and
    duplicate_ids == 0
    and
    len(forbidden_selected) == 0
    and
    fold1_train == 33
    and
    fold1_test == 33
    and
    fold2_train == 66
    and
    fold2_test == 33
    and
    len(final_feature_cols) > 0
)

print(
    "\nSaved final modelling dataset:"
)

print(
    OUTPUT_FILE
)

print(
    "\nSaved feature list:"
)

print(
    FEATURE_LIST_FILE
)

print(
    "\nSaved missingness report:"
)

print(
    MISSINGNESS_FILE
)

print(
    "\nSaved temporal fold assignments:"
)

print(
    FOLD_FILE
)

if pass_checks:

    print(
        "\nPASS: final modelling dataset is ready "
        "for temporal model training."
    )

else:

    print(
        "\nCHECK REQUIRED: review diagnostics "
        "before training models."
    )

print(
    "\n" + "=" * 72
)

print("DONE")

print("=" * 72)
