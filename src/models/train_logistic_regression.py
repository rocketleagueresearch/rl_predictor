from pathlib import Path
import json
import math
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

TABLES_DIR = (
    RESULTS_DIR
    / "tables"
)

MODELS_DIR = (
    PROJECT_ROOT
    / "src"
    / "models"
)

INPUT_FILE = (
    PROCESSED_DIR
    / "final_modelling_dataset.csv"
)

FEATURE_LIST_FILE = (
    PROCESSED_DIR
    / "final_modelling_feature_list.csv"
)

PREDICTIONS_FILE = (
    TABLES_DIR
    / "logistic_regression_predictions.csv"
)

FOLD_METRICS_FILE = (
    TABLES_DIR
    / "logistic_regression_fold_metrics.csv"
)

POOLED_METRICS_FILE = (
    TABLES_DIR
    / "logistic_regression_pooled_metrics.csv"
)

CONFUSION_FILE = (
    TABLES_DIR
    / "logistic_regression_confusion_matrices.csv"
)

COEFFICIENTS_FILE = (
    TABLES_DIR
    / "logistic_regression_coefficients.csv"
)

SUMMARY_FILE = (
    TABLES_DIR
    / "logistic_regression_summary.txt"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42
DECISION_THRESHOLD = 0.50

# Conservative baseline regularisation.
# We do not tune hyperparameters yet because the dataset is small,
# and we want a clean baseline before comparing more complex models.
MODEL_KWARGS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "liblinear",
    "max_iter": 5000,
    "random_state": RANDOM_STATE,
}


# ============================================================
# HELPERS
# ============================================================

def divider(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def metric_dict(y_true, y_prob, threshold=0.50):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    out = {
        "n": int(len(y_true)),
        "positive_rate": float(
            np.mean(y_true)
        ),
        "predicted_positive_rate": float(
            np.mean(y_pred)
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "log_loss": float(
            log_loss(
                y_true,
                y_prob,
                labels=[
                    0,
                    1,
                ],
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_true,
                y_prob,
            )
        ),
    }

    if len(
        np.unique(
            y_true
        )
    ) == 2:
        out[
            "roc_auc"
        ] = float(
            roc_auc_score(
                y_true,
                y_prob,
            )
        )
    else:
        out[
            "roc_auc"
        ] = np.nan

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
        ],
    ).ravel()

    out.update(
        {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
    )

    return out


def build_pipeline(feature_cols):
    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                feature_cols,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = LogisticRegression(
        **MODEL_KWARGS
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


def fit_and_evaluate_fold(
    df,
    feature_cols,
    fold_name,
    role_col,
):
    train_mask = (
        df[
            role_col
        ]
        ==
        "train"
    )

    test_mask = (
        df[
            role_col
        ]
        ==
        "test"
    )

    train = df.loc[
        train_mask
    ].copy()

    test = df.loc[
        test_mask
    ].copy()

    if len(
        train
    ) == 0:
        raise ValueError(
            f"{fold_name}: no training rows."
        )

    if len(
        test
    ) == 0:
        raise ValueError(
            f"{fold_name}: no test rows."
        )

    X_train = train[
        feature_cols
    ].copy()

    y_train = train[
        "target_team_a_win"
    ].astype(int)

    X_test = test[
        feature_cols
    ].copy()

    y_test = test[
        "target_team_a_win"
    ].astype(int)

    pipeline = build_pipeline(
        feature_cols
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    y_prob = pipeline.predict_proba(
        X_test
    )[:, 1]

    y_pred = (
        y_prob
        >=
        DECISION_THRESHOLD
    ).astype(int)

    metrics = metric_dict(
        y_test,
        y_prob,
        threshold=DECISION_THRESHOLD,
    )

    metrics[
        "fold"
    ] = fold_name

    metrics[
        "train_n"
    ] = int(
        len(
            train
        )
    )

    metrics[
        "test_n"
    ] = int(
        len(
            test
        )
    )

    metrics[
        "train_positive_rate"
    ] = float(
        y_train.mean()
    )

    preds = test[
        [
            "chronological_index",
            "series_id",
            "tournament_name",
            "series_start_utc",
            "team_a",
            "team_b",
            "target_team_a_win",
        ]
    ].copy()

    preds[
        "fold"
    ] = fold_name

    preds[
        "predicted_probability_team_a_win"
    ] = y_prob

    preds[
        "predicted_class"
    ] = y_pred

    preds[
        "correct"
    ] = (
        preds[
            "predicted_class"
        ]
        ==
        preds[
            "target_team_a_win"
        ]
    ).astype(int)

    # Extract coefficients after preprocessing.
    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    transformed_names = preprocessor.get_feature_names_out()

    model = pipeline.named_steps[
        "model"
    ]

    coefs = model.coef_[0]

    coef_df = pd.DataFrame(
        {
            "fold": fold_name,
            "feature": transformed_names,
            "coefficient": coefs,
            "abs_coefficient": np.abs(
                coefs
            ),
        }
    )

    coef_df = coef_df.sort_values(
        "abs_coefficient",
        ascending=False,
    ).reset_index(
        drop=True
    )

    return (
        pipeline,
        metrics,
        preds,
        coef_df,
    )


# ============================================================
# LOAD DATA
# ============================================================

divider("LOGISTIC REGRESSION TEMPORAL BASELINE")

TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Modelling dataset not found:\n{INPUT_FILE}"
    )

if not FEATURE_LIST_FILE.exists():
    raise FileNotFoundError(
        f"Feature list not found:\n{FEATURE_LIST_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE
)

feature_report = pd.read_csv(
    FEATURE_LIST_FILE
)

feature_cols = (
    feature_report.loc[
        feature_report[
            "model_predictor"
        ]
        ==
        True,
        "feature",
    ]
    .astype(str)
    .tolist()
)

missing_features = [
    col
    for col in feature_cols
    if col not in df.columns
]

if missing_features:
    raise ValueError(
        "Features listed but absent from modelling dataset: "
        + ", ".join(
            missing_features
        )
    )

print(
    f"Rows loaded: "
    f"{len(df)}"
)

print(
    f"Predictor features: "
    f"{len(feature_cols)}"
)

print(
    f"Target positives: "
    f"{int(df['target_team_a_win'].sum())}"
)

print(
    f"Target negatives: "
    f"{int((df['target_team_a_win'] == 0).sum())}"
)


# ============================================================
# VALIDATE TEMPORAL FOLDS
# ============================================================

fold_specs = [
    (
        "Fold 1: Open 1 -> Open 2",
        "fold1_role",
    ),
    (
        "Fold 2: Open 1+2 -> Open 3",
        "fold2_role",
    ),
]

for fold_name, role_col in fold_specs:

    if role_col not in df.columns:
        raise ValueError(
            f"Missing fold column: {role_col}"
        )

    train_n = int(
        (
            df[
                role_col
            ]
            ==
            "train"
        ).sum()
    )

    test_n = int(
        (
            df[
                role_col
            ]
            ==
            "test"
        ).sum()
    )

    print(
        f"{fold_name}: "
        f"train={train_n}, test={test_n}"
    )


# ============================================================
# TRAIN + EVALUATE
# ============================================================

all_metrics = []
all_predictions = []
all_coefficients = []

fitted_models = {}

for fold_name, role_col in fold_specs:

    divider(
        fold_name
    )

    (
        pipeline,
        metrics,
        predictions,
        coefficients,
    ) = fit_and_evaluate_fold(
        df=df,
        feature_cols=feature_cols,
        fold_name=fold_name,
        role_col=role_col,
    )

    fitted_models[
        fold_name
    ] = pipeline

    all_metrics.append(
        metrics
    )

    all_predictions.append(
        predictions
    )

    all_coefficients.append(
        coefficients
    )

    print(
        f"Train rows: "
        f"{metrics['train_n']}"
    )

    print(
        f"Test rows: "
        f"{metrics['test_n']}"
    )

    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"Log loss: "
        f"{metrics['log_loss']:.4f}"
    )

    print(
        f"Brier score: "
        f"{metrics['brier_score']:.4f}"
    )

    print(
        "Confusion matrix "
        f"(TN={metrics['tn']}, "
        f"FP={metrics['fp']}, "
        f"FN={metrics['fn']}, "
        f"TP={metrics['tp']})"
    )


# ============================================================
# POOLED OUT-OF-SAMPLE EVALUATION
# ============================================================

divider("POOLED OUT-OF-SAMPLE RESULTS")

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True,
)

pooled_metrics = metric_dict(
    predictions_df[
        "target_team_a_win"
    ],
    predictions_df[
        "predicted_probability_team_a_win"
    ],
    threshold=DECISION_THRESHOLD,
)

pooled_metrics[
    "fold"
] = "Pooled OOS: Open 2 + Open 3"

print(
    f"Out-of-sample rows: "
    f"{pooled_metrics['n']}"
)

print(
    f"Accuracy: "
    f"{pooled_metrics['accuracy']:.4f}"
)

print(
    f"Precision: "
    f"{pooled_metrics['precision']:.4f}"
)

print(
    f"Recall: "
    f"{pooled_metrics['recall']:.4f}"
)

print(
    f"F1: "
    f"{pooled_metrics['f1']:.4f}"
)

print(
    f"ROC-AUC: "
    f"{pooled_metrics['roc_auc']:.4f}"
)

print(
    f"Log loss: "
    f"{pooled_metrics['log_loss']:.4f}"
)

print(
    f"Brier score: "
    f"{pooled_metrics['brier_score']:.4f}"
)

print(
    "Confusion matrix "
    f"(TN={pooled_metrics['tn']}, "
    f"FP={pooled_metrics['fp']}, "
    f"FN={pooled_metrics['fn']}, "
    f"TP={pooled_metrics['tp']})"
)


# ============================================================
# BASELINE COMPARISON
# ============================================================

# Majority-class benchmark derived ONLY from each training fold.
# This is useful context for the MSc evaluation.

divider("TRAINING-FOLD MAJORITY BASELINE")

baseline_rows = []

for fold_name, role_col in fold_specs:

    train = df.loc[
        df[
            role_col
        ]
        ==
        "train"
    ]

    test = df.loc[
        df[
            role_col
        ]
        ==
        "test"
    ]

    train_positive_rate = float(
        train[
            "target_team_a_win"
        ].mean()
    )

    majority_class = int(
        train_positive_rate
        >=
        0.5
    )

    baseline_pred = np.full(
        len(
            test
        ),
        majority_class,
        dtype=int,
    )

    baseline_accuracy = float(
        accuracy_score(
            test[
                "target_team_a_win"
            ].astype(int),
            baseline_pred,
        )
    )

    baseline_rows.append(
        {
            "fold": fold_name,
            "train_positive_rate": train_positive_rate,
            "majority_class": majority_class,
            "test_accuracy": baseline_accuracy,
        }
    )

    print(
        f"{fold_name}: "
        f"training majority class={majority_class}, "
        f"test accuracy={baseline_accuracy:.4f}"
    )


# ============================================================
# SAVE TABLES
# ============================================================

metrics_df = pd.DataFrame(
    all_metrics
)

pooled_df = pd.DataFrame(
    [
        pooled_metrics
    ]
)

coefficients_df = pd.concat(
    all_coefficients,
    ignore_index=True,
)

confusion_rows = []

for metrics in all_metrics + [
    pooled_metrics
]:

    confusion_rows.append(
        {
            "fold": metrics[
                "fold"
            ],
            "tn": metrics[
                "tn"
            ],
            "fp": metrics[
                "fp"
            ],
            "fn": metrics[
                "fn"
            ],
            "tp": metrics[
                "tp"
            ],
        }
    )

confusion_df = pd.DataFrame(
    confusion_rows
)

predictions_df = predictions_df.sort_values(
    "chronological_index"
).reset_index(
    drop=True
)

metrics_df.to_csv(
    FOLD_METRICS_FILE,
    index=False,
)

pooled_df.to_csv(
    POOLED_METRICS_FILE,
    index=False,
)

predictions_df.to_csv(
    PREDICTIONS_FILE,
    index=False,
)

confusion_df.to_csv(
    CONFUSION_FILE,
    index=False,
)

coefficients_df.to_csv(
    COEFFICIENTS_FILE,
    index=False,
)


# ============================================================
# TEXT SUMMARY
# ============================================================

summary_lines = []

summary_lines.append(
    "LOGISTIC REGRESSION TEMPORAL BASELINE"
)

summary_lines.append(
    "=" * 72
)

summary_lines.append(
    f"Rows: {len(df)}"
)

summary_lines.append(
    f"Predictor features: {len(feature_cols)}"
)

summary_lines.append(
    ""
)

for metrics in all_metrics:

    summary_lines.append(
        metrics[
            "fold"
        ]
    )

    summary_lines.append(
        f"  Train n: {metrics['train_n']}"
    )

    summary_lines.append(
        f"  Test n: {metrics['test_n']}"
    )

    summary_lines.append(
        f"  Accuracy: {metrics['accuracy']:.4f}"
    )

    summary_lines.append(
        f"  Precision: {metrics['precision']:.4f}"
    )

    summary_lines.append(
        f"  Recall: {metrics['recall']:.4f}"
    )

    summary_lines.append(
        f"  F1: {metrics['f1']:.4f}"
    )

    summary_lines.append(
        f"  ROC-AUC: {metrics['roc_auc']:.4f}"
    )

    summary_lines.append(
        f"  Log loss: {metrics['log_loss']:.4f}"
    )

    summary_lines.append(
        f"  Brier score: {metrics['brier_score']:.4f}"
    )

    summary_lines.append(
        ""
    )

summary_lines.append(
    "POOLED OOS: OPEN 2 + OPEN 3"
)

summary_lines.append(
    f"  n: {pooled_metrics['n']}"
)

summary_lines.append(
    f"  Accuracy: {pooled_metrics['accuracy']:.4f}"
)

summary_lines.append(
    f"  Precision: {pooled_metrics['precision']:.4f}"
)

summary_lines.append(
    f"  Recall: {pooled_metrics['recall']:.4f}"
)

summary_lines.append(
    f"  F1: {pooled_metrics['f1']:.4f}"
)

summary_lines.append(
    f"  ROC-AUC: {pooled_metrics['roc_auc']:.4f}"
)

summary_lines.append(
    f"  Log loss: {pooled_metrics['log_loss']:.4f}"
)

summary_lines.append(
    f"  Brier score: {pooled_metrics['brier_score']:.4f}"
)

SUMMARY_FILE.write_text(
    "\n".join(
        summary_lines
    ),
    encoding="utf-8",
)


# ============================================================
# TOP COEFFICIENTS
# ============================================================

divider("LARGEST ABSOLUTE COEFFICIENTS")

for fold_name, _ in fold_specs:

    print(
        f"\n{fold_name}"
    )

    top = (
        coefficients_df.loc[
            coefficients_df[
                "fold"
            ]
            ==
            fold_name
        ]
        .sort_values(
            "abs_coefficient",
            ascending=False,
        )
        .head(
            12
        )
    )

    print(
        top[
            [
                "feature",
                "coefficient",
                "abs_coefficient",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

divider("FILES SAVED")

print(
    PREDICTIONS_FILE
)

print(
    FOLD_METRICS_FILE
)

print(
    POOLED_METRICS_FILE
)

print(
    CONFUSION_FILE
)

print(
    COEFFICIENTS_FILE
)

print(
    SUMMARY_FILE
)

print(
    "\nPASS: logistic regression baseline completed "
    "with temporal out-of-sample evaluation."
)

print(
    "\n" + "=" * 72
)

print("DONE")

print("=" * 72)
