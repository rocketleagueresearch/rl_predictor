from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILES = {
    "Logistic Regression": {
        "predictions": TABLES_DIR / "logistic_regression_predictions.csv",
        "fold_metrics": TABLES_DIR / "logistic_regression_fold_metrics.csv",
        "pooled_metrics": TABLES_DIR / "logistic_regression_pooled_metrics.csv",
    },
    "Random Forest": {
        "predictions": TABLES_DIR / "random_forest_predictions.csv",
        "fold_metrics": TABLES_DIR / "random_forest_fold_metrics.csv",
        "pooled_metrics": TABLES_DIR / "random_forest_pooled_metrics.csv",
    },
    "Neural Network": {
        "predictions": TABLES_DIR / "neural_network_predictions.csv",
        "fold_metrics": TABLES_DIR / "neural_network_fold_metrics.csv",
        "pooled_metrics": TABLES_DIR / "neural_network_pooled_metrics.csv",
    },
}

COMBINED_POOLED_FILE = TABLES_DIR / "combined_model_pooled_metrics.csv"
COMBINED_FOLD_FILE = TABLES_DIR / "combined_model_fold_metrics.csv"
CALIBRATION_FILE = TABLES_DIR / "combined_model_calibration.csv"
ERROR_OVERLAP_FILE = TABLES_DIR / "combined_model_error_overlap.csv"
SERIES_COMPARISON_FILE = TABLES_DIR / "combined_model_series_predictions.csv"
SUMMARY_FILE = TABLES_DIR / "combined_model_evaluation_summary.txt"

ROC_FIGURE = FIGURES_DIR / "model_roc_curves.png"
CALIBRATION_FIGURE = FIGURES_DIR / "model_calibration_curves.png"
METRIC_FIGURE = FIGURES_DIR / "model_metric_comparison.png"
CONFUSION_FIGURE = FIGURES_DIR / "model_confusion_matrices.png"


# ============================================================
# HELPERS
# ============================================================

def divider(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def standardise_prediction_columns(df, model_name):
    required = [
        "series_id",
        "tournament_name",
        "series_start_utc",
        "target_team_a_win",
        "predicted_probability_team_a_win",
        "predicted_class",
        "fold",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"{model_name}: prediction file missing columns: "
            + ", ".join(missing)
        )

    out = df.copy()

    out["target_team_a_win"] = (
        pd.to_numeric(out["target_team_a_win"], errors="raise").astype(int)
    )

    out["predicted_probability_team_a_win"] = pd.to_numeric(
        out["predicted_probability_team_a_win"],
        errors="raise",
    )

    out["predicted_class"] = (
        pd.to_numeric(out["predicted_class"], errors="raise").astype(int)
    )

    out["correct"] = (
        out["predicted_class"] == out["target_team_a_win"]
    ).astype(int)

    out["model"] = model_name

    return out


def calculate_metrics(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob, labels=[0, 1]),
        "brier_score": brier_score_loss(y_true, y_prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# LOAD AND VALIDATE
# ============================================================

divider("FINAL MODEL COMPARISON")

prediction_frames = {}
pooled_rows = []
fold_rows = []

for model_name, paths in MODEL_FILES.items():

    for path_name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{model_name} {path_name} file not found:\n{path}"
            )

    pred = pd.read_csv(paths["predictions"])
    pred = standardise_prediction_columns(pred, model_name)

    prediction_frames[model_name] = pred

    pooled_metrics = calculate_metrics(
        pred["target_team_a_win"],
        pred["predicted_probability_team_a_win"],
    )
    pooled_metrics["model"] = model_name
    pooled_metrics["scope"] = "Pooled OOS"
    pooled_rows.append(pooled_metrics)

    for fold_name, group in pred.groupby("fold", sort=False):
        metrics = calculate_metrics(
            group["target_team_a_win"],
            group["predicted_probability_team_a_win"],
        )
        metrics["model"] = model_name
        metrics["fold"] = fold_name
        fold_rows.append(metrics)

    print(
        f"{model_name}: rows={len(pred)}, "
        f"accuracy={pooled_metrics['accuracy']:.4f}, "
        f"ROC-AUC={pooled_metrics['roc_auc']:.4f}"
    )


# ============================================================
# ALIGNMENT CHECK
# ============================================================

divider("PREDICTION ALIGNMENT AUDIT")

base_model = next(iter(prediction_frames))
base = prediction_frames[base_model].sort_values(
    ["series_start_utc", "series_id"]
).reset_index(drop=True)

for model_name, pred in prediction_frames.items():
    aligned = pred.sort_values(
        ["series_start_utc", "series_id"]
    ).reset_index(drop=True)

    if len(aligned) != len(base):
        raise ValueError(
            f"{model_name}: prediction row count differs from {base_model}."
        )

    same_ids = aligned["series_id"].equals(base["series_id"])
    same_targets = aligned["target_team_a_win"].equals(
        base["target_team_a_win"]
    )

    print(
        f"{model_name}: same series order={same_ids}, "
        f"same targets={same_targets}"
    )

    if not same_ids or not same_targets:
        raise ValueError(
            f"{model_name}: predictions do not align with other models."
        )


# ============================================================
# SAVE COMBINED METRICS
# ============================================================

pooled_df = pd.DataFrame(pooled_rows)

metric_order = [
    "model",
    "scope",
    "n",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "log_loss",
    "brier_score",
    "tn",
    "fp",
    "fn",
    "tp",
]

pooled_df = pooled_df[metric_order]

fold_df = pd.DataFrame(fold_rows)

fold_order = [
    "model",
    "fold",
    "n",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "log_loss",
    "brier_score",
    "tn",
    "fp",
    "fn",
    "tp",
]

fold_df = fold_df[fold_order]

pooled_df.to_csv(
    COMBINED_POOLED_FILE,
    index=False,
)

fold_df.to_csv(
    COMBINED_FOLD_FILE,
    index=False,
)


# ============================================================
# CALIBRATION
# ============================================================

divider("CALIBRATION")

calibration_rows = []

for model_name, pred in prediction_frames.items():

    y_true = pred["target_team_a_win"].to_numpy()
    y_prob = pred["predicted_probability_team_a_win"].to_numpy()

    prob_true, prob_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=5,
        strategy="quantile",
    )

    for bin_number, (mean_pred, observed) in enumerate(
        zip(prob_pred, prob_true),
        start=1,
    ):
        calibration_rows.append(
            {
                "model": model_name,
                "bin": bin_number,
                "mean_predicted_probability": mean_pred,
                "observed_team_a_win_rate": observed,
            }
        )

    print(
        f"{model_name}: 5-bin quantile calibration calculated."
    )

calibration_df = pd.DataFrame(calibration_rows)

calibration_df.to_csv(
    CALIBRATION_FILE,
    index=False,
)


# ============================================================
# SERIES-LEVEL COMPARISON + ERROR OVERLAP
# ============================================================

divider("ERROR OVERLAP")

series_comparison = base[
    [
        "series_id",
        "tournament_name",
        "series_start_utc",
        "team_a",
        "team_b",
        "target_team_a_win",
        "fold",
    ]
].copy()

for model_name, pred in prediction_frames.items():

    aligned = pred.sort_values(
        ["series_start_utc", "series_id"]
    ).reset_index(drop=True)

    prefix = (
        model_name.lower()
        .replace(" ", "_")
    )

    series_comparison[
        f"{prefix}_probability"
    ] = aligned[
        "predicted_probability_team_a_win"
    ]

    series_comparison[
        f"{prefix}_prediction"
    ] = aligned[
        "predicted_class"
    ]

    series_comparison[
        f"{prefix}_correct"
    ] = aligned[
        "correct"
    ]

correct_cols = [
    col for col in series_comparison.columns
    if col.endswith("_correct")
]

series_comparison[
    "models_correct"
] = series_comparison[
    correct_cols
].sum(axis=1)

series_comparison[
    "models_wrong"
] = (
    len(correct_cols)
    -
    series_comparison["models_correct"]
)

series_comparison.to_csv(
    SERIES_COMPARISON_FILE,
    index=False,
)

overlap_summary = (
    series_comparison[
        "models_wrong"
    ]
    .value_counts()
    .sort_index()
    .rename_axis("number_of_models_wrong")
    .reset_index(name="series_count")
)

overlap_summary[
    "percentage_of_oos_series"
] = (
    overlap_summary["series_count"]
    /
    len(series_comparison)
    *
    100
)

overlap_summary.to_csv(
    ERROR_OVERLAP_FILE,
    index=False,
)

print(overlap_summary.to_string(index=False))

all_wrong = series_comparison.loc[
    series_comparison["models_wrong"] == 3
].copy()

print(
    f"\nSeries all three models got wrong: {len(all_wrong)}"
)

if len(all_wrong) > 0:
    print(
        all_wrong[
            [
                "series_id",
                "tournament_name",
                "team_a",
                "team_b",
                "target_team_a_win",
            ]
        ].to_string(index=False)
    )


# ============================================================
# ROC FIGURE
# ============================================================

plt.figure(figsize=(8, 6))

for model_name, pred in prediction_frames.items():

    y_true = pred["target_team_a_win"].to_numpy()
    y_prob = pred["predicted_probability_team_a_win"].to_numpy()

    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob,
    )

    auc = roc_auc_score(
        y_true,
        y_prob,
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC={auc:.3f})",
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Chance",
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Pooled Out-of-Sample ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig(
    ROC_FIGURE,
    dpi=300,
)
plt.close()


# ============================================================
# CALIBRATION FIGURE
# ============================================================

plt.figure(figsize=(8, 6))

for model_name in MODEL_FILES:

    group = calibration_df.loc[
        calibration_df["model"] == model_name
    ]

    plt.plot(
        group["mean_predicted_probability"],
        group["observed_team_a_win_rate"],
        marker="o",
        label=model_name,
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration",
)

plt.xlabel("Mean Predicted Probability")
plt.ylabel("Observed Team A Win Rate")
plt.title("Pooled Out-of-Sample Calibration")
plt.legend()
plt.tight_layout()
plt.savefig(
    CALIBRATION_FIGURE,
    dpi=300,
)
plt.close()


# ============================================================
# METRIC COMPARISON FIGURE
# ============================================================

display_metrics = [
    "accuracy",
    "f1",
    "roc_auc",
]

metric_plot = pooled_df.set_index("model")[
    display_metrics
]

ax = metric_plot.plot(
    kind="bar",
    figsize=(9, 6),
)

ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.set_title("Pooled Out-of-Sample Classification Metrics")
ax.legend(
    [
        "Accuracy",
        "F1",
        "ROC-AUC",
    ]
)

plt.xticks(
    rotation=0
)
plt.tight_layout()
plt.savefig(
    METRIC_FIGURE,
    dpi=300,
)
plt.close()


# ============================================================
# CONFUSION MATRICES FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(12, 4),
)

for ax, model_name in zip(
    axes,
    MODEL_FILES.keys(),
):
    row = pooled_df.loc[
        pooled_df["model"] == model_name
    ].iloc[0]

    matrix = np.array(
        [
            [row["tn"], row["fp"]],
            [row["fn"], row["tp"]],
        ]
    )

    image = ax.imshow(
        matrix,
    )

    ax.set_title(model_name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Team B", "Team A"])
    ax.set_yticklabels(["Team B", "Team A"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
            )

fig.suptitle(
    "Pooled Out-of-Sample Confusion Matrices"
)

plt.tight_layout()
plt.savefig(
    CONFUSION_FIGURE,
    dpi=300,
)
plt.close()


# ============================================================
# SUMMARY
# ============================================================

divider("FINAL POOLED MODEL RANKING")

ranking_accuracy = pooled_df.sort_values(
    "accuracy",
    ascending=False,
)[
    ["model", "accuracy"]
]

ranking_auc = pooled_df.sort_values(
    "roc_auc",
    ascending=False,
)[
    ["model", "roc_auc"]
]

ranking_logloss = pooled_df.sort_values(
    "log_loss",
    ascending=True,
)[
    ["model", "log_loss"]
]

ranking_brier = pooled_df.sort_values(
    "brier_score",
    ascending=True,
)[
    ["model", "brier_score"]
]

print("\nAccuracy:")
print(ranking_accuracy.to_string(index=False))

print("\nROC-AUC:")
print(ranking_auc.to_string(index=False))

print("\nLog loss:")
print(ranking_logloss.to_string(index=False))

print("\nBrier score:")
print(ranking_brier.to_string(index=False))

summary_lines = [
    "FINAL MODEL COMPARISON",
    "=" * 78,
    "",
    "POOLED OUT-OF-SAMPLE METRICS",
    pooled_df.to_string(index=False),
    "",
    "ERROR OVERLAP",
    overlap_summary.to_string(index=False),
    "",
    f"Series all three models got wrong: {len(all_wrong)}",
    "",
    "RANKING BY ACCURACY",
    ranking_accuracy.to_string(index=False),
    "",
    "RANKING BY ROC-AUC",
    ranking_auc.to_string(index=False),
    "",
    "RANKING BY LOG LOSS",
    ranking_logloss.to_string(index=False),
    "",
    "RANKING BY BRIER SCORE",
    ranking_brier.to_string(index=False),
]

SUMMARY_FILE.write_text(
    "\n".join(summary_lines),
    encoding="utf-8",
)


# ============================================================
# FINAL
# ============================================================

divider("FILES SAVED")

for path in [
    COMBINED_POOLED_FILE,
    COMBINED_FOLD_FILE,
    CALIBRATION_FILE,
    ERROR_OVERLAP_FILE,
    SERIES_COMPARISON_FILE,
    SUMMARY_FILE,
    ROC_FIGURE,
    CALIBRATION_FIGURE,
    METRIC_FIGURE,
    CONFUSION_FIGURE,
]:
    print(path)

print()
print(
    "PASS: final model comparison, calibration, ROC, "
    "confusion-matrix and error-overlap analysis completed."
)

print()
print("=" * 78)
print("DONE")
print("=" * 78)
