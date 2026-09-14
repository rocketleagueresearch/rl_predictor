from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42
N_BOOTSTRAPS = 10000
CI_LEVEL = 0.95

ROOT_DIR = Path(__file__).resolve().parents[2]
TABLES_DIR = ROOT_DIR / "results" / "tables"

MODEL_FILES = {
    "Logistic Regression": (
        TABLES_DIR / "logistic_regression_predictions.csv"
    ),
    "Random Forest": (
        TABLES_DIR / "random_forest_predictions.csv"
    ),
    "Neural Network": (
        TABLES_DIR / "neural_network_predictions.csv"
    ),
}

MODEL_CI_FILE = (
    TABLES_DIR
    / "bootstrap_model_confidence_intervals.csv"
)

PAIRWISE_FILE = (
    TABLES_DIR
    / "bootstrap_model_pairwise_comparisons.csv"
)

SUMMARY_FILE = (
    TABLES_DIR
    / "bootstrap_model_uncertainty_summary.txt"
)


# ============================================================
# HELPERS
# ============================================================

def divider(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def find_column(df, candidates, description):
    """
    Return the first matching column from a list of candidate
    names.

    Matching is case-sensitive first, then case-insensitive.
    """

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    lower_lookup = {
        str(column).lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        key = candidate.lower()

        if key in lower_lookup:
            return lower_lookup[key]

    raise ValueError(
        f"Could not identify {description} column.\n"
        f"Tried: {candidates}\n"
        f"Available columns:\n{list(df.columns)}"
    )


def standardise_predictions(df, model_name):
    """
    Standardise each model prediction file to:

    series_id
    tournament
    fold
    target
    probability
    prediction
    """

    series_col = find_column(
        df,
        [
            "series_id",
            "series",
            "match_id",
        ],
        "series identifier",
    )

    target_col = find_column(
        df,
        [
            "target_team_a_win",
            "target",
            "y_true",
            "actual",
        ],
        "target",
    )

    probability_col = find_column(
        df,
        [
            "predicted_probability",
            "predicted_probability_team_a_win",
            "probability_team_a_win",
            "team_a_win_probability",
            "predicted_proba",
            "y_probability",
            "y_prob",
            "probability",
        ],
        "predicted probability",
    )

    prediction_col = find_column(
        df,
        [
            "predicted_class",
            "prediction",
            "y_pred",
            "predicted",
        ],
        "predicted class",
    )

    fold_col = find_column(
        df,
        [
            "fold",
            "test_fold",
            "evaluation_fold",
        ],
        "fold",
    )

    tournament_col = None

    for candidate in [
        "tournament",
        "open",
        "event",
    ]:
        if candidate in df.columns:
            tournament_col = candidate
            break

    out = pd.DataFrame()

    out["series_id"] = (
        df[series_col]
        .astype(str)
        .str.strip()
    )

    out["target"] = pd.to_numeric(
        df[target_col],
        errors="raise",
    ).astype(int)

    out["probability"] = pd.to_numeric(
        df[probability_col],
        errors="raise",
    ).astype(float)

    out["prediction"] = pd.to_numeric(
        df[prediction_col],
        errors="raise",
    ).astype(int)

    out["fold"] = (
        df[fold_col]
        .astype(str)
        .str.strip()
    )

    if tournament_col is not None:
        out["tournament"] = (
            df[tournament_col]
            .astype(str)
            .str.strip()
        )
    else:
        out["tournament"] = ""

    if out["series_id"].duplicated().any():
        duplicates = out.loc[
            out["series_id"].duplicated(
                keep=False
            ),
            "series_id",
        ].tolist()

        raise ValueError(
            f"{model_name}: duplicate series IDs found:\n"
            f"{duplicates}"
        )

    if not out["target"].isin([0, 1]).all():
        raise ValueError(
            f"{model_name}: target contains values "
            "other than 0 and 1."
        )

    if not out["prediction"].isin([0, 1]).all():
        raise ValueError(
            f"{model_name}: predicted class contains values "
            "other than 0 and 1."
        )

    if (
        out["probability"].isna().any()
        or not out["probability"]
        .between(0, 1)
        .all()
    ):
        raise ValueError(
            f"{model_name}: predicted probabilities must "
            "all be between 0 and 1."
        )

    return out


def calculate_metrics(
    y_true,
    probabilities,
    predictions,
):
    """
    Calculate the five final evaluation metrics.
    """

    result = {
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "log_loss": log_loss(
            y_true,
            probabilities,
            labels=[0, 1],
        ),
        "brier_score": brier_score_loss(
            y_true,
            probabilities,
        ),
    }

    # ROC-AUC cannot be calculated if a bootstrap
    # sample happens to contain only one target class.
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = roc_auc_score(
            y_true,
            probabilities,
        )
    else:
        result["roc_auc"] = np.nan

    return result


def percentile_interval(values):
    """
    Percentile bootstrap confidence interval.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan

    alpha = 1.0 - CI_LEVEL

    lower = np.quantile(
        values,
        alpha / 2.0,
    )

    upper = np.quantile(
        values,
        1.0 - alpha / 2.0,
    )

    return float(lower), float(upper)


def pairwise_better_difference(
    metric,
    first_value,
    second_value,
):
    """
    Return a difference where POSITIVE always means that
    the first model performed better.

    Higher-is-better:
        accuracy
        f1
        roc_auc

    Lower-is-better:
        log_loss
        brier_score
    """

    if metric in {
        "accuracy",
        "f1",
        "roc_auc",
    }:
        return first_value - second_value

    if metric in {
        "log_loss",
        "brier_score",
    }:
        return second_value - first_value

    raise ValueError(
        f"Unknown metric: {metric}"
    )


# ============================================================
# LOAD PREDICTIONS
# ============================================================

divider("BOOTSTRAP MODEL UNCERTAINTY ANALYSIS")

print(
    f"Bootstrap repetitions: {N_BOOTSTRAPS}"
)
print(
    f"Random seed: {RANDOM_SEED}"
)
print(
    f"Confidence level: {CI_LEVEL:.0%}"
)

prediction_frames = {}

for model_name, path in MODEL_FILES.items():

    if not path.exists():
        raise FileNotFoundError(
            f"{model_name} prediction file not found:\n"
            f"{path}"
        )

    raw = pd.read_csv(path)

    standardised = standardise_predictions(
        raw,
        model_name,
    )

    standardised = (
        standardised
        .sort_values("series_id")
        .reset_index(drop=True)
    )

    prediction_frames[
        model_name
    ] = standardised

    print(
        f"{model_name}: "
        f"{len(standardised)} prediction rows loaded."
    )


# ============================================================
# VERIFY EXACT PAIRED ALIGNMENT
# ============================================================

divider("PAIRED ALIGNMENT CHECK")

model_names = list(
    prediction_frames.keys()
)

reference_name = model_names[0]

reference = prediction_frames[
    reference_name
]

for model_name in model_names[1:]:

    current = prediction_frames[
        model_name
    ]

    if len(current) != len(reference):
        raise ValueError(
            f"{model_name}: prediction row count "
            f"differs from {reference_name}."
        )

    same_series = (
        current["series_id"].tolist()
        == reference["series_id"].tolist()
    )

    same_target = np.array_equal(
        current["target"].to_numpy(),
        reference["target"].to_numpy(),
    )

    same_fold = (
        current["fold"].tolist()
        == reference["fold"].tolist()
    )

    print(
        f"{model_name}: "
        f"same series={same_series}, "
        f"same targets={same_target}, "
        f"same folds={same_fold}"
    )

    if not (
        same_series
        and same_target
        and same_fold
    ):
        raise ValueError(
            f"{model_name}: predictions are not aligned "
            "with the other models."
        )


n_series = len(reference)

print(
    f"\nAligned out-of-sample series: {n_series}"
)

print(
    "\nRows by fold:"
)

print(
    reference["fold"]
    .value_counts(sort=False)
    .to_string()
)


# ============================================================
# ORIGINAL POOLED METRICS
# ============================================================

divider("ORIGINAL POOLED METRICS")

original_metrics = {}

for model_name, df in prediction_frames.items():

    metrics = calculate_metrics(
        df["target"].to_numpy(),
        df["probability"].to_numpy(),
        df["prediction"].to_numpy(),
    )

    original_metrics[
        model_name
    ] = metrics

    print(
        f"\n{model_name}"
    )

    for metric, value in metrics.items():
        print(
            f"  {metric}: {value:.6f}"
        )


# ============================================================
# PREPARE FOLD-STRATIFIED PAIRED BOOTSTRAP
# ============================================================

divider("RUNNING PAIRED BOOTSTRAP")

rng = np.random.default_rng(
    RANDOM_SEED
)

fold_values = (
    reference["fold"]
    .drop_duplicates()
    .tolist()
)

fold_indices = {}

for fold in fold_values:

    indices = np.flatnonzero(
        reference["fold"].to_numpy()
        == fold
    )

    fold_indices[fold] = indices

    print(
        f"{fold}: {len(indices)} rows"
    )


metric_names = [
    "accuracy",
    "f1",
    "roc_auc",
    "log_loss",
    "brier_score",
]

bootstrap_results = {
    model_name: {
        metric: []
        for metric in metric_names
    }
    for model_name in model_names
}


# Store paired model differences for every
# bootstrap repetition.
pair_names = []

for i in range(len(model_names)):
    for j in range(
        i + 1,
        len(model_names),
    ):
        pair_names.append(
            (
                model_names[i],
                model_names[j],
            )
        )

pairwise_distributions = {
    pair: {
        metric: []
        for metric in metric_names
    }
    for pair in pair_names
}


for bootstrap_number in range(
    N_BOOTSTRAPS
):

    sampled_parts = []

    # Resample WITH replacement independently inside
    # each temporal test fold.
    #
    # This preserves the original 33/33 fold contribution
    # to the pooled out-of-sample evaluation while still
    # allowing individual series to vary across replicates.
    for fold in fold_values:

        indices = fold_indices[fold]

        sampled = rng.choice(
            indices,
            size=len(indices),
            replace=True,
        )

        sampled_parts.append(
            sampled
        )

    sampled_indices = np.concatenate(
        sampled_parts
    )

    iteration_metrics = {}

    for model_name in model_names:

        df = prediction_frames[
            model_name
        ]

        sample = df.iloc[
            sampled_indices
        ]

        metrics = calculate_metrics(
            sample["target"].to_numpy(),
            sample["probability"].to_numpy(),
            sample["prediction"].to_numpy(),
        )

        iteration_metrics[
            model_name
        ] = metrics

        for metric in metric_names:
            bootstrap_results[
                model_name
            ][metric].append(
                metrics[metric]
            )

    # Because every model uses the exact same sampled
    # series indices, these differences are paired.
    for first_model, second_model in pair_names:

        for metric in metric_names:

            first_value = (
                iteration_metrics[
                    first_model
                ][metric]
            )

            second_value = (
                iteration_metrics[
                    second_model
                ][metric]
            )

            if (
                np.isfinite(first_value)
                and np.isfinite(second_value)
            ):
                difference = (
                    pairwise_better_difference(
                        metric,
                        first_value,
                        second_value,
                    )
                )
            else:
                difference = np.nan

            pairwise_distributions[
                (
                    first_model,
                    second_model,
                )
            ][metric].append(
                difference
            )


print(
    "\nBootstrap resampling complete."
)


# ============================================================
# MODEL-SPECIFIC CONFIDENCE INTERVALS
# ============================================================

divider("MODEL CONFIDENCE INTERVALS")

ci_rows = []

for model_name in model_names:

    for metric in metric_names:

        values = np.asarray(
            bootstrap_results[
                model_name
            ][metric],
            dtype=float,
        )

        valid_values = values[
            np.isfinite(values)
        ]

        lower, upper = (
            percentile_interval(
                valid_values
            )
        )

        original_value = (
            original_metrics[
                model_name
            ][metric]
        )

        bootstrap_mean = float(
            np.mean(valid_values)
        )

        bootstrap_std = float(
            np.std(
                valid_values,
                ddof=1,
            )
        )

        ci_rows.append(
            {
                "model": model_name,
                "metric": metric,
                "original_value": (
                    original_value
                ),
                "bootstrap_mean": (
                    bootstrap_mean
                ),
                "bootstrap_std": (
                    bootstrap_std
                ),
                "ci_level": CI_LEVEL,
                "ci_lower": lower,
                "ci_upper": upper,
                "valid_bootstrap_samples": (
                    len(valid_values)
                ),
                "total_bootstrap_samples": (
                    N_BOOTSTRAPS
                ),
            }
        )

        print(
            f"{model_name:22s} "
            f"{metric:12s} "
            f"value={original_value:.4f} "
            f"95% CI="
            f"[{lower:.4f}, {upper:.4f}]"
        )


ci_df = pd.DataFrame(
    ci_rows
)


# ============================================================
# PAIRED MODEL COMPARISONS
# ============================================================

divider("PAIRED MODEL COMPARISONS")

pairwise_rows = []

for first_model, second_model in pair_names:

    print(
        f"\n{first_model} vs {second_model}"
    )

    for metric in metric_names:

        distribution = np.asarray(
            pairwise_distributions[
                (
                    first_model,
                    second_model,
                )
            ][metric],
            dtype=float,
        )

        valid = distribution[
            np.isfinite(distribution)
        ]

        lower, upper = (
            percentile_interval(
                valid
            )
        )

        original_difference = (
            pairwise_better_difference(
                metric,
                original_metrics[
                    first_model
                ][metric],
                original_metrics[
                    second_model
                ][metric],
            )
        )

        probability_first_better = float(
            np.mean(
                valid > 0
            )
        )

        probability_second_better = float(
            np.mean(
                valid < 0
            )
        )

        probability_tie = float(
            np.mean(
                valid == 0
            )
        )

        # Simple two-sided bootstrap tail probability.
        #
        # This is included as descriptive supporting
        # evidence rather than treated as a definitive
        # classical significance test.
        p_two_sided = min(
            1.0,
            2.0
            * min(
                float(
                    np.mean(
                        valid <= 0
                    )
                ),
                float(
                    np.mean(
                        valid >= 0
                    )
                ),
            ),
        )

        excludes_zero = (
            lower > 0
            or upper < 0
        )

        if lower > 0:
            ci_direction = (
                f"{first_model} better"
            )
        elif upper < 0:
            ci_direction = (
                f"{second_model} better"
            )
        else:
            ci_direction = (
                "uncertain / overlaps zero"
            )

        pairwise_rows.append(
            {
                "model_1": first_model,
                "model_2": second_model,
                "metric": metric,
                "difference_definition": (
                    "positive = model_1 better"
                ),
                "original_better_oriented_difference": (
                    original_difference
                ),
                "bootstrap_mean_difference": float(
                    np.mean(valid)
                ),
                "ci_level": CI_LEVEL,
                "ci_lower": lower,
                "ci_upper": upper,
                "ci_excludes_zero": (
                    excludes_zero
                ),
                "ci_direction": (
                    ci_direction
                ),
                "probability_model_1_better": (
                    probability_first_better
                ),
                "probability_model_2_better": (
                    probability_second_better
                ),
                "probability_tie": (
                    probability_tie
                ),
                "bootstrap_two_sided_tail_p": (
                    p_two_sided
                ),
                "valid_bootstrap_samples": (
                    len(valid)
                ),
            }
        )

        print(
            f"  {metric:12s} "
            f"diff={original_difference:+.4f} "
            f"95% CI="
            f"[{lower:+.4f}, {upper:+.4f}] "
            f"P(first better)="
            f"{probability_first_better:.3f}"
        )


pairwise_df = pd.DataFrame(
    pairwise_rows
)


# ============================================================
# SAVE RESULTS
# ============================================================

divider("SAVING RESULTS")

ci_df.to_csv(
    MODEL_CI_FILE,
    index=False,
)

pairwise_df.to_csv(
    PAIRWISE_FILE,
    index=False,
)


summary_lines = [
    "BOOTSTRAP MODEL UNCERTAINTY ANALYSIS",
    "",
    (
        f"Bootstrap repetitions: "
        f"{N_BOOTSTRAPS}"
    ),
    (
        f"Random seed: "
        f"{RANDOM_SEED}"
    ),
    (
        f"Confidence level: "
        f"{CI_LEVEL:.0%}"
    ),
    (
        f"Out-of-sample series: "
        f"{n_series}"
    ),
    "",
    (
        "Method: paired, fold-stratified percentile "
        "bootstrap."
    ),
    (
        "The same resampled series indices were used "
        "for all three models."
    ),
    (
        "Resampling occurred separately within each "
        "temporal test fold, preserving each fold's "
        "contribution to the pooled evaluation."
    ),
    (
        "No model was retrained and no features, "
        "hyperparameters, thresholds, or preprocessing "
        "were changed."
    ),
    "",
    "MODEL 95% CONFIDENCE INTERVALS",
    "",
]

for _, row in ci_df.iterrows():

    summary_lines.append(
        f"{row['model']} | "
        f"{row['metric']} | "
        f"value={row['original_value']:.6f} | "
        f"95% CI=["
        f"{row['ci_lower']:.6f}, "
        f"{row['ci_upper']:.6f}]"
    )


summary_lines.extend(
    [
        "",
        "PAIRED MODEL COMPARISONS",
        "",
        (
            "For every metric below, a positive "
            "difference means model_1 performed better."
        ),
        (
            "For accuracy, F1 and ROC-AUC, higher values "
            "are better."
        ),
        (
            "For log loss and Brier score, lower values "
            "are better, so their pairwise differences "
            "are direction-adjusted."
        ),
        "",
    ]
)

for _, row in pairwise_df.iterrows():

    summary_lines.append(
        f"{row['model_1']} vs "
        f"{row['model_2']} | "
        f"{row['metric']} | "
        f"diff="
        f"{row['original_better_oriented_difference']:+.6f} | "
        f"95% CI=["
        f"{row['ci_lower']:+.6f}, "
        f"{row['ci_upper']:+.6f}] | "
        f"P(model_1 better)="
        f"{row['probability_model_1_better']:.4f} | "
        f"{row['ci_direction']}"
    )


summary_lines.extend(
    [
        "",
        "INTERPRETATION NOTE",
        "",
        (
            "These bootstrap intervals quantify sampling "
            "uncertainty in the observed 66-series "
            "out-of-sample evaluation."
        ),
        (
            "They should not be interpreted as proving "
            "performance on every future Rocket League "
            "event."
        ),
        (
            "The paired comparisons are particularly "
            "useful because all three models were "
            "evaluated on the exact same series."
        ),
    ]
)


SUMMARY_FILE.write_text(
    "\n".join(summary_lines),
    encoding="utf-8",
)


print(
    f"\nSaved:\n{MODEL_CI_FILE}"
)

print(
    f"\nSaved:\n{PAIRWISE_FILE}"
)

print(
    f"\nSaved:\n{SUMMARY_FILE}"
)


# ============================================================
# FINAL VALIDATION
# ============================================================

divider("FINAL VALIDATION")

expected_ci_rows = (
    len(model_names)
    * len(metric_names)
)

expected_pairwise_rows = (
    len(pair_names)
    * len(metric_names)
)

checks = {
    "66 pooled OOS rows": (
        n_series == 66
    ),
    "15 model CI rows": (
        len(ci_df)
        == expected_ci_rows
    ),
    "15 paired comparison rows": (
        len(pairwise_df)
        == expected_pairwise_rows
    ),
    "No missing CI bounds": (
        ci_df[
            [
                "ci_lower",
                "ci_upper",
            ]
        ]
        .notna()
        .all()
        .all()
    ),
    "No missing pairwise CI bounds": (
        pairwise_df[
            [
                "ci_lower",
                "ci_upper",
            ]
        ]
        .notna()
        .all()
        .all()
    ),
}

all_passed = True

for check_name, passed in checks.items():

    print(
        f"{check_name}: "
        f"{'PASS' if passed else 'FAIL'}"
    )

    if not passed:
        all_passed = False


if all_passed:

    print(
        "\nPASS: bootstrap uncertainty analysis "
        "completed successfully."
    )

else:

    raise ValueError(
        "One or more final bootstrap validation "
        "checks failed."
    )