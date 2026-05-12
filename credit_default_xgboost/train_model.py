from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise SystemExit(
        "xgboost is required. Install project dependencies with: "
        "pip install -r credit_default_xgboost/requirements.txt"
    ) from exc


PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
REPORT_DIR = PROJECT_DIR / "reports"
PDP_DIR = REPORT_DIR / "pdp"
TARGET_COLUMN = "credit_card_default"
RANDOM_STATE = 42
MAX_FEATURES = 18
CORRELATION_LIMIT = 0.60


FEATURE_CATALOG: Dict[str, Dict[str, str]] = {
    "credit_score": {
        "label": "Credit score",
        "group": "Credit quality",
        "input_type": "number",
        "rationale": "Summarizes historical credit behavior in one familiar measure.",
    },
    "credit_utilization": {
        "label": "Credit utilization",
        "group": "Balance pressure",
        "input_type": "percent",
        "rationale": "Shows how much of the available credit line is currently being used.",
    },
    "balance_to_limit_6m_avg": {
        "label": "6-month average balance-to-limit",
        "group": "Balance pressure",
        "input_type": "percent",
        "rationale": "Measures sustained utilization pressure over the recent period.",
    },
    "max_pay_delay": {
        "label": "Worst recent payment delay",
        "group": "Payment behavior",
        "input_type": "integer",
        "rationale": "Captures the most severe recent repayment delay.",
    },
    "num_late_payments": {
        "label": "Number of late payments",
        "group": "Payment behavior",
        "input_type": "integer",
        "rationale": "Counts how often recent payments were late.",
    },
    "payment_consistency_score": {
        "label": "Payment consistency score",
        "group": "Payment behavior",
        "input_type": "number",
        "rationale": "Higher consistency indicates more reliable repayment behavior.",
    },
    "pay_to_bill_ratio": {
        "label": "Payment-to-bill ratio",
        "group": "Payment behavior",
        "input_type": "percent",
        "rationale": "Compares recent payments with billed amounts.",
    },
    "avg_pay_amt": {
        "label": "Average payment amount",
        "group": "Payment behavior",
        "input_type": "currency",
        "rationale": "Shows typical recent repayment amount.",
    },
    "payment_to_income_ratio": {
        "label": "Payment-to-income ratio",
        "group": "Capacity",
        "input_type": "number",
        "rationale": "Compares repayment behavior with income capacity.",
    },
    "debt_to_income_ratio": {
        "label": "Debt-to-income ratio",
        "group": "Capacity",
        "input_type": "number",
        "rationale": "Measures total debt burden relative to income.",
    },
    "monthly_obligation_ratio": {
        "label": "Monthly obligation ratio",
        "group": "Capacity",
        "input_type": "number",
        "rationale": "Shows how much recurring obligations consume available income.",
    },
    "annual_income": {
        "label": "Annual income",
        "group": "Capacity",
        "input_type": "currency",
        "rationale": "Represents borrower repayment capacity.",
    },
    "income_stability_index": {
        "label": "Income stability index",
        "group": "Capacity",
        "input_type": "number",
        "rationale": "Captures the steadiness of income over time.",
    },
    "credit_limit_to_income": {
        "label": "Credit limit-to-income",
        "group": "Capacity",
        "input_type": "number",
        "rationale": "Compares approved credit exposure with income.",
    },
    "spend_to_income_ratio": {
        "label": "Spend-to-income ratio",
        "group": "Capacity",
        "input_type": "number",
        "rationale": "Shows spending pressure relative to income.",
    },
    "spend_to_limit_ratio": {
        "label": "Spend-to-limit ratio",
        "group": "Balance pressure",
        "input_type": "number",
        "rationale": "Compares spending level with the credit limit.",
    },
    "current_balance": {
        "label": "Current balance",
        "group": "Balance pressure",
        "input_type": "currency",
        "rationale": "Measures the current outstanding balance.",
    },
    "limit_bal": {
        "label": "Credit limit",
        "group": "Account profile",
        "input_type": "currency",
        "rationale": "Represents total approved credit exposure.",
    },
    "num_credit_inquiries_6m": {
        "label": "Recent credit inquiries",
        "group": "Credit seeking",
        "input_type": "integer",
        "rationale": "Frequent recent credit searches may indicate liquidity pressure.",
    },
    "num_open_accounts": {
        "label": "Open accounts",
        "group": "Credit profile",
        "input_type": "integer",
        "rationale": "Shows breadth of current credit obligations.",
    },
    "oldest_account_years": {
        "label": "Oldest account age",
        "group": "Credit profile",
        "input_type": "number",
        "rationale": "Longer account history can indicate more established credit behavior.",
    },
    "num_delinquencies_2yr": {
        "label": "Delinquencies in last 2 years",
        "group": "Credit quality",
        "input_type": "integer",
        "rationale": "Counts recent serious repayment issues.",
    },
    "num_public_records": {
        "label": "Public records",
        "group": "Credit quality",
        "input_type": "integer",
        "rationale": "Captures public credit events that may indicate elevated risk.",
    },
    "bankruptcy_flag": {
        "label": "Bankruptcy flag",
        "group": "Credit quality",
        "input_type": "binary",
        "rationale": "Indicates whether a bankruptcy signal is present.",
    },
    "debt_collection_flag": {
        "label": "Debt collection flag",
        "group": "Credit quality",
        "input_type": "binary",
        "rationale": "Indicates whether a collection signal is present.",
    },
    "returned_payment_count": {
        "label": "Returned payment count",
        "group": "Payment behavior",
        "input_type": "integer",
        "rationale": "Counts failed or returned payments.",
    },
    "cash_advance_usage_pct": {
        "label": "Cash advance usage",
        "group": "Liquidity pressure",
        "input_type": "percent",
        "rationale": "Cash advance reliance can indicate short-term liquidity stress.",
    },
    "cash_advance_count_6m": {
        "label": "Cash advances in 6 months",
        "group": "Liquidity pressure",
        "input_type": "integer",
        "rationale": "Counts recent cash advance use.",
    },
    "customer_service_calls_6m": {
        "label": "Customer service calls in 6 months",
        "group": "Account behavior",
        "input_type": "integer",
        "rationale": "Frequent service contact can indicate account stress or disputes.",
    },
    "dispute_count_1yr": {
        "label": "Disputes in 1 year",
        "group": "Account behavior",
        "input_type": "integer",
        "rationale": "Captures recent account dispute activity.",
    },
    "months_since_last_delinquency": {
        "label": "Months since last delinquency",
        "group": "Credit quality",
        "input_type": "integer",
        "rationale": "Shows recency of the last delinquency event.",
    },
    "autopay_enrolled": {
        "label": "Autopay enrolled",
        "group": "Payment behavior",
        "input_type": "binary",
        "rationale": "Autopay can reduce missed-payment risk.",
    },
    "min_pay_flag": {
        "label": "Minimum payment flag",
        "group": "Payment behavior",
        "input_type": "binary",
        "rationale": "Indicates reliance on minimum payments.",
    },
    "repeat_late_payer_flag": {
        "label": "Repeat late payer flag",
        "group": "Payment behavior",
        "input_type": "binary",
        "rationale": "Identifies repeated late-payment behavior.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an explainable XGBoost default model.")
    parser.add_argument(
        "--data",
        default=str(PROJECT_DIR.parent / "credit.csv"),
        help="Path to credit.csv. Defaults to ../credit.csv from this project folder.",
    )
    parser.add_argument("--target", default=TARGET_COLUMN, help="Binary target column name.")
    parser.add_argument("--max-features", type=int, default=MAX_FEATURES)
    parser.add_argument("--corr-limit", type=float, default=CORRELATION_LIMIT)
    parser.add_argument("--validation-size", type=float, default=0.30)
    parser.add_argument(
        "--search-method",
        choices=("bayesian", "grid"),
        default="bayesian",
        help="Hyperparameter search method. Defaults to Bayesian optimization.",
    )
    parser.add_argument(
        "--bayes-init",
        type=int,
        default=18,
        help="Initial random XGBoost configurations for Bayesian optimization.",
    )
    parser.add_argument(
        "--bayes-iterations",
        type=int,
        default=72,
        help="Bayesian optimization iterations after the initial random phase.",
    )
    parser.add_argument(
        "--bayes-candidates",
        type=int,
        default=600,
        help="Candidate configurations sampled at each Bayesian optimization step.",
    )
    return parser.parse_args()


def load_data(path: Path, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find data file: {path}")

    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available columns: {list(df.columns)}")

    y = df[target].astype(int)
    X = df.drop(columns=[target])
    if sorted(y.unique()) != [0, 1]:
        raise ValueError(f"Target '{target}' must be binary with values 0 and 1.")
    return X, y


def candidate_features(columns: Iterable[str]) -> List[str]:
    available = set(columns)
    return [feature for feature in FEATURE_CATALOG if feature in available]


def score_candidate_features(X: pd.DataFrame, y: pd.Series, features: List[str]) -> pd.DataFrame:
    X_numeric = X[features].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X_numeric),
        columns=features,
        index=X_numeric.index,
    )

    mi_scores = mutual_info_classif(X_imputed, y, discrete_features="auto", random_state=RANDOM_STATE)
    rows = []
    for feature, mi_score in zip(features, mi_scores):
        values = X_imputed[feature]
        if values.nunique() <= 1:
            auc_lift = 0.0
        else:
            auc = roc_auc_score(y, values)
            auc_lift = abs(auc - 0.5)
        rows.append(
            {
                "feature": feature,
                "mutual_information": float(mi_score),
                "univariate_auc_lift": float(auc_lift),
            }
        )

    scores = pd.DataFrame(rows)
    scores["mi_rank_score"] = scores["mutual_information"].rank(pct=True)
    scores["auc_rank_score"] = scores["univariate_auc_lift"].rank(pct=True)
    scores["selection_score"] = 0.65 * scores["mi_rank_score"] + 0.35 * scores["auc_rank_score"]
    return scores.sort_values("selection_score", ascending=False)


def select_uncorrelated_features(
    X: pd.DataFrame,
    y: pd.Series,
    max_features: int,
    corr_limit: float,
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    features = candidate_features(X.columns)
    scores = score_candidate_features(X, y, features)
    corr = X[features].apply(pd.to_numeric, errors="coerce").corr().abs().fillna(0)

    selected: List[str] = []
    rejected_rows = []

    for feature in scores["feature"]:
        if len(selected) >= max_features:
            break

        correlated_with = [
            selected_feature
            for selected_feature in selected
            if corr.loc[feature, selected_feature] >= corr_limit
        ]
        if correlated_with:
            rejected_rows.append(
                {
                    "feature": feature,
                    "reason": "correlation_limit",
                    "correlated_with": ", ".join(correlated_with),
                    "max_abs_corr": float(max(corr.loc[feature, correlated_with])),
                }
            )
            continue
        selected.append(feature)

    return selected, scores, pd.DataFrame(rejected_rows)


XGB_PARAM_SPACE = {
    "n_estimators": ("int", 150, 900),
    "max_depth": ("int", 2, 5),
    "learning_rate": ("log", 0.015, 0.18),
    "min_child_weight": ("log", 0.5, 12.0),
    "subsample": ("float", 0.65, 1.0),
    "colsample_bytree": ("float", 0.65, 1.0),
    "reg_lambda": ("log", 0.2, 25.0),
    "reg_alpha": ("log", 0.0001, 5.0),
    "gamma": ("float", 0.0, 3.0),
}


def decode_params(encoded: np.ndarray) -> Dict[str, float]:
    params: Dict[str, float] = {}
    for value, (name, spec) in zip(encoded, XGB_PARAM_SPACE.items()):
        kind, low, high = spec
        value = float(np.clip(value, 0.0, 1.0))
        if kind == "int":
            params[name] = int(round(low + value * (high - low)))
        elif kind == "log":
            params[name] = float(math.exp(math.log(low) + value * (math.log(high) - math.log(low))))
        else:
            params[name] = float(low + value * (high - low))
    return params


def parameter_signature(params: Dict[str, float]) -> Tuple[object, ...]:
    signature = []
    for name, spec in XGB_PARAM_SPACE.items():
        kind = spec[0]
        value = params[name]
        if kind == "int":
            signature.append((name, int(value)))
        else:
            signature.append((name, round(float(value), 6)))
    return tuple(signature)


def normal_pdf(values: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * values**2) / math.sqrt(2.0 * math.pi)


def normal_cdf(values: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(values / math.sqrt(2.0)))


def expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    best_score: float,
    exploration: float = 0.0005,
) -> np.ndarray:
    std = np.maximum(std, 1e-9)
    improvement = mean - best_score - exploration
    z = improvement / std
    return improvement * normal_cdf(z) + std * normal_pdf(z)


def fit_evaluate_xgboost(
    params: Dict[str, float],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scale_pos_weight: float,
) -> Tuple[XGBClassifier, Dict[str, float]]:
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        **params,
    )
    model.fit(X_train, y_train, verbose=False)
    proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, proba)
    pr_auc = average_precision_score(y_val, proba)
    metrics = {
        "roc_auc": float(auc),
        "gini": float(2 * auc - 1),
        "average_precision": float(pr_auc),
        "log_loss": float(log_loss(y_val, proba)),
    }
    return model, metrics


def bayesian_parameter_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scale_pos_weight: float,
    initial_points: int,
    iterations: int,
    candidate_count: int,
) -> Tuple[XGBClassifier, pd.DataFrame]:
    rng = np.random.RandomState(RANDOM_STATE)
    dimensions = len(XGB_PARAM_SPACE)
    rows = []
    observations_X = []
    observations_y = []
    tried = set()
    best_model = None
    best_auc = -np.inf

    total_steps = max(1, initial_points + iterations)
    for step in range(total_steps):
        if step < initial_points or len(observations_X) < 6:
            encoded = rng.rand(dimensions)
            params = decode_params(encoded)
            while parameter_signature(params) in tried:
                encoded = rng.rand(dimensions)
                params = decode_params(encoded)
            phase = "random_initialization"
            acquisition = np.nan
        else:
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                length_scale=np.ones(dimensions),
                length_scale_bounds=(1e-2, 1e2),
                nu=2.5,
            ) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-8, 1e-2))
            optimizer = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                random_state=RANDOM_STATE,
                n_restarts_optimizer=1,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                optimizer.fit(np.array(observations_X), np.array(observations_y))

            candidates = rng.rand(max(candidate_count, 1), dimensions)
            candidate_params = [decode_params(candidate) for candidate in candidates]
            valid_indices = [
                index
                for index, params_item in enumerate(candidate_params)
                if parameter_signature(params_item) not in tried
            ]
            if not valid_indices:
                encoded = rng.rand(dimensions)
                params = decode_params(encoded)
                phase = "fallback_random"
                acquisition = np.nan
            else:
                valid_candidates = candidates[valid_indices]
                mean, std = optimizer.predict(valid_candidates, return_std=True)
                ei = expected_improvement(mean, std, best_auc)
                best_candidate_position = int(np.argmax(ei))
                encoded = valid_candidates[best_candidate_position]
                params = decode_params(encoded)
                phase = "bayesian_expected_improvement"
                acquisition = float(ei[best_candidate_position])

        tried.add(parameter_signature(params))
        model, model_metrics = fit_evaluate_xgboost(
            params,
            X_train,
            y_train,
            X_val,
            y_val,
            scale_pos_weight,
        )
        auc = model_metrics["roc_auc"]
        observations_X.append(encoded)
        observations_y.append(auc)

        row = {
            "search_method": "bayesian",
            "search_step": step + 1,
            "search_phase": phase,
            "acquisition_expected_improvement": acquisition,
            **params,
            **model_metrics,
        }
        rows.append(row)

        if auc > best_auc:
            best_auc = auc
            best_model = model
            print(f"Bayesian search step {step + 1}/{total_steps}: new best ROC-AUC={auc:.6f}")

    if best_model is None:
        raise RuntimeError("Bayesian search did not train a model.")

    leaderboard = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    return best_model, leaderboard


def grid_parameter_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scale_pos_weight: float,
) -> Tuple[XGBClassifier, pd.DataFrame]:
    param_grid = {
        "n_estimators": [180, 300, 450],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.03, 0.06],
        "min_child_weight": [1, 4],
        "subsample": [0.85],
        "colsample_bytree": [0.85],
        "reg_lambda": [1.0, 4.0],
        "reg_alpha": [0.0],
        "gamma": [0.0],
    }

    rows = []
    best_model = None
    best_auc = -np.inf
    for step, params in enumerate(ParameterGrid(param_grid), start=1):
        model, model_metrics = fit_evaluate_xgboost(
            params,
            X_train,
            y_train,
            X_val,
            y_val,
            scale_pos_weight,
        )
        rows.append(
            {
                "search_method": "grid",
                "search_step": step,
                "search_phase": "grid",
                "acquisition_expected_improvement": np.nan,
                **params,
                **model_metrics,
            }
        )
        if model_metrics["roc_auc"] > best_auc:
            best_auc = model_metrics["roc_auc"]
            best_model = model

    if best_model is None:
        raise RuntimeError("Grid search did not train a model.")

    leaderboard = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    return best_model, leaderboard


def train_xgboost_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: List[str],
    search_method: str,
    bayes_init: int,
    bayes_iterations: int,
    bayes_candidates: int,
) -> Tuple[XGBClassifier, pd.DataFrame]:
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / max(pos_count, 1)

    if search_method == "bayesian":
        best_model, leaderboard = bayesian_parameter_search(
            X_train,
            y_train,
            X_val,
            y_val,
            scale_pos_weight,
            initial_points=bayes_init,
            iterations=bayes_iterations,
            candidate_count=bayes_candidates,
        )
    elif search_method == "grid":
        best_model, leaderboard = grid_parameter_search(
            X_train,
            y_train,
            X_val,
            y_val,
            scale_pos_weight,
        )
    else:
        raise ValueError(f"Unsupported search method: {search_method}")

    best_model.get_booster().feature_names = feature_names
    return best_model, leaderboard


def choose_threshold(y_true: pd.Series, proba: np.ndarray) -> Tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.linspace(0.05, 0.95, 181):
        pred = (proba >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "f1": float(f1_score(y_true, pred, zero_division=0)),
            }
        )

    threshold_table = pd.DataFrame(rows)
    best_row = threshold_table.sort_values(["f1", "recall"], ascending=False).iloc[0]
    return float(best_row["threshold"]), threshold_table


def ks_statistic(y_true: pd.Series, proba: np.ndarray) -> float:
    scored = pd.DataFrame({"y": y_true.values, "proba": proba})
    positives = scored.loc[scored["y"] == 1, "proba"].sort_values()
    negatives = scored.loc[scored["y"] == 0, "proba"].sort_values()
    grid = np.sort(scored["proba"].unique())
    pos_cdf = np.searchsorted(positives.values, grid, side="right") / max(len(positives), 1)
    neg_cdf = np.searchsorted(negatives.values, grid, side="right") / max(len(negatives), 1)
    return float(np.max(np.abs(pos_cdf - neg_cdf)))


def feature_stats(df: pd.DataFrame, features: List[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for feature in features:
        values = pd.to_numeric(df[feature], errors="coerce")
        stats[feature] = {
            "min": float(values.min()),
            "p01": float(values.quantile(0.01)),
            "p05": float(values.quantile(0.05)),
            "p25": float(values.quantile(0.25)),
            "median": float(values.median()),
            "p75": float(values.quantile(0.75)),
            "p95": float(values.quantile(0.95)),
            "p99": float(values.quantile(0.99)),
            "max": float(values.max()),
        }
    return stats


def make_feature_metadata(features: List[str], train_df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    stats = feature_stats(train_df, features)
    metadata: Dict[str, Dict[str, object]] = {}
    for feature in features:
        item = dict(FEATURE_CATALOG[feature])
        item["name"] = feature
        item["stats"] = stats[feature]
        metadata[feature] = item
    return metadata


def generate_pdp_reports(
    model: XGBClassifier,
    X_val_df: pd.DataFrame,
    selected_features: List[str],
    feature_metadata: Dict[str, Dict[str, object]],
    top_n: int = 8,
) -> pd.DataFrame:
    PDP_DIR.mkdir(parents=True, exist_ok=True)
    importances = pd.Series(model.feature_importances_, index=selected_features).sort_values(ascending=False)
    top_features = list(importances.head(top_n).index)
    rows = []

    for feature in top_features:
        values = X_val_df[feature].astype(float)
        unique_values = np.sort(values.dropna().unique())
        if len(unique_values) <= 8:
            grid = unique_values
        else:
            grid = np.unique(np.percentile(values, np.linspace(5, 95, 25)))

        averages = []
        for value in grid:
            scenario = X_val_df.copy()
            scenario[feature] = value
            averages.append(float(model.predict_proba(scenario[selected_features])[:, 1].mean()))

        label = str(feature_metadata[feature]["label"])
        file_stem = feature.replace("/", "_")
        plot_path = PDP_DIR / f"pdp_{file_stem}.png"

        plt.figure(figsize=(8, 4.8))
        plt.plot(grid, averages, color="#0f766e", linewidth=2.5)
        plt.scatter(grid, averages, color="#2458d3", s=18)
        plt.title(f"Partial Dependence: {label}")
        plt.xlabel(label)
        plt.ylabel("Average predicted default probability")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=160)
        plt.close()

        rows.append(
            {
                "feature": feature,
                "label": label,
                "plot_file": str(plot_path.relative_to(PROJECT_DIR)),
                "min_grid_value": float(grid[0]),
                "max_grid_value": float(grid[-1]),
                "pdp_at_min": float(averages[0]),
                "pdp_at_max": float(averages[-1]),
                "direction": "increases risk" if averages[-1] > averages[0] else "decreases risk",
            }
        )

    return pd.DataFrame(rows)


def generate_validation_distribution_reports(
    validation_predictions: pd.DataFrame,
    threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    report_df = validation_predictions.copy()
    scores = report_df["predicted_default_probability"]

    quantiles = scores.describe(
        percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ).reset_index()
    quantiles.columns = ["statistic", "all_validation_records"]

    by_actual = []
    for actual_value, label in [(0, "actual_non_default"), (1, "actual_default")]:
        subset = report_df.loc[report_df["actual_default"] == actual_value, "predicted_default_probability"]
        summary = subset.describe(
            percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        )
        by_actual.append(summary.rename(label))
    quantile_summary = pd.concat([quantiles.set_index("statistic")] + by_actual, axis=1).reset_index()

    ranked = report_df.sort_values("predicted_default_probability", ascending=False).reset_index(drop=True)
    ranked["risk_decile"] = pd.qcut(
        ranked.index + 1,
        q=10,
        labels=list(range(1, 11)),
    ).astype(int)
    deciles = (
        ranked.groupby("risk_decile")
        .agg(
            records=("actual_default", "size"),
            defaults=("actual_default", "sum"),
            observed_default_rate=("actual_default", "mean"),
            average_predicted_probability=("predicted_default_probability", "mean"),
            min_predicted_probability=("predicted_default_probability", "min"),
            max_predicted_probability=("predicted_default_probability", "max"),
        )
        .reset_index()
        .sort_values("risk_decile")
    )
    total_records = float(deciles["records"].sum())
    total_defaults = float(deciles["defaults"].sum())
    portfolio_default_rate = total_defaults / total_records
    deciles["decile_population_pct"] = deciles["records"] / total_records
    deciles["decile_default_capture_rate"] = deciles["defaults"] / total_defaults
    deciles["decile_lift"] = deciles["observed_default_rate"] / portfolio_default_rate
    deciles["cumulative_records"] = deciles["records"].cumsum()
    deciles["cumulative_defaults"] = deciles["defaults"].cumsum()
    deciles["cumulative_population_pct"] = deciles["cumulative_records"] / total_records
    deciles["cumulative_default_capture_rate"] = deciles["cumulative_defaults"] / total_defaults
    deciles["cumulative_lift"] = (
        deciles["cumulative_default_capture_rate"] / deciles["cumulative_population_pct"]
    )

    plt.figure(figsize=(9, 5.4))
    bins = np.linspace(0, 1, 31)
    non_default_scores = report_df.loc[
        report_df["actual_default"] == 0,
        "predicted_default_probability",
    ]
    default_scores = report_df.loc[
        report_df["actual_default"] == 1,
        "predicted_default_probability",
    ]
    plt.hist(
        non_default_scores,
        bins=bins,
        alpha=0.62,
        density=True,
        color="#2458d3",
        label="Actual non-default",
    )
    plt.hist(
        default_scores,
        bins=bins,
        alpha=0.56,
        density=True,
        color="#b74434",
        label="Actual default",
    )
    plt.axvline(
        threshold,
        color="#111917",
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold:.3f}",
    )
    plt.title("Validation Predicted Default Probability Distribution")
    plt.xlabel("Predicted default probability")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.22)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "validation_score_distribution.png", dpi=170)
    plt.close()

    plt.figure(figsize=(9, 5.4))
    plt.plot(
        deciles["risk_decile"].to_numpy(),
        deciles["observed_default_rate"].to_numpy(),
        marker="o",
        linewidth=2.5,
        color="#b74434",
        label="Observed default rate",
    )
    plt.plot(
        deciles["risk_decile"].to_numpy(),
        deciles["average_predicted_probability"].to_numpy(),
        marker="o",
        linewidth=2.5,
        color="#0f766e",
        label="Average predicted probability",
    )
    plt.gca().invert_xaxis()
    plt.title("Validation Risk Deciles")
    plt.xlabel("Risk decile, 1 = highest predicted risk")
    plt.ylabel("Rate")
    plt.legend()
    plt.grid(True, alpha=0.22)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "validation_decile_lift.png", dpi=170)
    plt.close()

    plt.figure(figsize=(9, 5.4))
    x_values = deciles["cumulative_population_pct"].to_numpy()
    capture_values = deciles["cumulative_default_capture_rate"].to_numpy()
    plt.plot(
        x_values,
        capture_values,
        marker="o",
        linewidth=2.8,
        color="#0f766e",
        label="Model cumulative default capture",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=2,
        color="#626a73",
        label="Random selection baseline",
    )
    for _, row in deciles.iterrows():
        plt.annotate(
            f"D{int(row['risk_decile'])}",
            (
                float(row["cumulative_population_pct"]),
                float(row["cumulative_default_capture_rate"]),
            ),
            textcoords="offset points",
            xytext=(4, 5),
            fontsize=9,
        )
    plt.title("Validation Cumulative Default Capture By Risk Decile")
    plt.xlabel("Cumulative population reviewed")
    plt.ylabel("Cumulative defaults captured")
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    plt.legend()
    plt.grid(True, alpha=0.22)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "validation_decile_capture.png", dpi=170)
    plt.close()

    return quantile_summary, deciles


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_model_report(
    metrics: Dict[str, object],
    selected_features: List[str],
    feature_metadata: Dict[str, Dict[str, object]],
    rejected_features: pd.DataFrame,
    leaderboard: pd.DataFrame,
) -> None:
    lines = [
        "# Credit Default XGBoost Model Report",
        "",
        "## Hyperparameter Search",
        "",
        f"- Search method: {metrics.get('hyperparameter_search_method', 'unknown')}",
        f"- Search objective: {metrics.get('hyperparameter_search_objective', 'ROC-AUC')}",
        f"- Evaluated configurations: {metrics.get('hyperparameter_search_evaluations', 'unknown')}",
        "",
        "## Measurement Method",
        "",
        "Primary model measurement: **ROC-AUC**, translated to **Gini = 2 x AUC - 1**.",
        "",
        "Why: credit default scoring is a ranking/discrimination problem. ROC-AUC is threshold-independent, is robust for this 30% default-rate dataset, and maps to the Gini measure commonly used in credit-risk model monitoring. Precision-recall AUC, KS, F1, recall, precision, and confusion matrix are reported as supporting measures.",
        "",
        "## Validation Metrics",
        "",
    ]
    for key, value in metrics.items():
        if key == "confusion_matrix":
            continue
        lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "Confusion matrix order: actual rows [non-default, default] x predicted columns [non-default, default].",
            "",
            "```json",
            json.dumps(metrics["confusion_matrix"], indent=2),
            "```",
            "",
            "## Selected Features",
            "",
        ]
    )
    for feature in selected_features:
        item = feature_metadata[feature]
        lines.append(f"- **{item['label']}** (`{feature}`): {item['rationale']}")

    if not rejected_features.empty:
        lines.extend(["", "## Features Rejected By Correlation Rule", ""])
        for _, row in rejected_features.iterrows():
            lines.append(
                f"- `{row['feature']}` rejected because it correlated with "
                f"`{row['correlated_with']}` at {row['max_abs_corr']:.3f}."
            )

    lines.extend(["", "## Best XGBoost Parameter Sets By Validation ROC-AUC", "", "```text"])
    lines.append(leaderboard.head(10).to_string(index=False))
    lines.extend(["```", ""])
    (REPORT_DIR / "model_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PDP_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_data(Path(args.data), args.target)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.validation_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    selected_features, feature_scores, rejected_features = select_uncorrelated_features(
        X_train_raw,
        y_train,
        max_features=args.max_features,
        corr_limit=args.corr_limit,
    )
    if len(selected_features) == 0:
        raise RuntimeError("Feature selection returned no usable features.")

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train_raw[selected_features])
    X_val = imputer.transform(X_val_raw[selected_features])
    X_train_df = pd.DataFrame(X_train, columns=selected_features, index=X_train_raw.index)
    X_val_df = pd.DataFrame(X_val, columns=selected_features, index=X_val_raw.index)

    model, leaderboard = train_xgboost_models(
        X_train_df,
        y_train,
        X_val_df,
        y_val,
        feature_names=selected_features,
        search_method=args.search_method,
        bayes_init=args.bayes_init,
        bayes_iterations=args.bayes_iterations,
        bayes_candidates=args.bayes_candidates,
    )

    val_proba = model.predict_proba(X_val_df[selected_features])[:, 1]
    threshold, threshold_table = choose_threshold(y_val, val_proba)
    val_pred = (val_proba >= threshold).astype(int)
    auc = roc_auc_score(y_val, val_proba)
    pr_auc = average_precision_score(y_val, val_proba)
    conf_matrix = confusion_matrix(y_val, val_pred).tolist()
    best_search_row = leaderboard.iloc[0].to_dict()
    best_hyperparameters = {
        name: (
            int(best_search_row[name])
            if XGB_PARAM_SPACE[name][0] == "int"
            else round(float(best_search_row[name]), 8)
        )
        for name in XGB_PARAM_SPACE
        if name in best_search_row
    }

    metrics: Dict[str, object] = {
        "target": args.target,
        "train_rows": int(len(X_train_raw)),
        "validation_rows": int(len(X_val_raw)),
        "validation_size": float(args.validation_size),
        "positive_default_rate_train": float(y_train.mean()),
        "positive_default_rate_validation": float(y_val.mean()),
        "primary_measurement_method": "ROC-AUC / Gini",
        "hyperparameter_search_method": args.search_method,
        "hyperparameter_search_objective": "maximize validation ROC-AUC",
        "hyperparameter_search_evaluations": int(len(leaderboard)),
        "best_hyperparameters": best_hyperparameters,
        "roc_auc": round(float(auc), 6),
        "gini": round(float(2 * auc - 1), 6),
        "average_precision_pr_auc": round(float(pr_auc), 6),
        "ks_statistic": round(ks_statistic(y_val, val_proba), 6),
        "log_loss": round(float(log_loss(y_val, val_proba)), 6),
        "classification_threshold": round(float(threshold), 6),
        "precision_at_threshold": round(float(precision_score(y_val, val_pred, zero_division=0)), 6),
        "recall_at_threshold": round(float(recall_score(y_val, val_pred, zero_division=0)), 6),
        "f1_at_threshold": round(float(f1_score(y_val, val_pred, zero_division=0)), 6),
        "confusion_matrix": conf_matrix,
    }

    feature_metadata = make_feature_metadata(selected_features, X_train_raw)
    pdp_summary = generate_pdp_reports(model, X_val_df, selected_features, feature_metadata)

    validation_predictions = pd.DataFrame(
        {
            "actual_default": y_val.values,
            "predicted_default_probability": val_proba,
            "predicted_default": val_pred,
        },
        index=X_val_raw.index,
    )
    quantile_summary, deciles = generate_validation_distribution_reports(
        validation_predictions,
        threshold,
    )

    artifacts = {
        "model": model,
        "imputer": imputer,
        "selected_features": selected_features,
        "feature_metadata": feature_metadata,
        "threshold": threshold,
        "metrics": metrics,
        "model_leaderboard": leaderboard,
    }
    joblib.dump(artifacts, ARTIFACT_DIR / "default_xgboost_artifacts.joblib")

    write_json(ARTIFACT_DIR / "metrics.json", metrics)
    write_json(
        ARTIFACT_DIR / "selected_features.json",
        {
            "selected_features": selected_features,
            "feature_metadata": feature_metadata,
            "correlation_limit": args.corr_limit,
            "max_features": args.max_features,
        },
    )
    feature_scores.to_csv(REPORT_DIR / "feature_selection_scores.csv", index=False)
    rejected_features.to_csv(REPORT_DIR / "correlation_rejected_features.csv", index=False)
    leaderboard.to_csv(REPORT_DIR / "model_leaderboard.csv", index=False)
    threshold_table.to_csv(REPORT_DIR / "threshold_analysis.csv", index=False)
    validation_predictions.to_csv(REPORT_DIR / "validation_predictions.csv", index=True)
    quantile_summary.to_csv(REPORT_DIR / "validation_score_quantiles.csv", index=False)
    deciles.to_csv(REPORT_DIR / "validation_score_deciles.csv", index=False)
    pdp_summary.to_csv(REPORT_DIR / "partial_dependence_summary.csv", index=False)
    write_model_report(metrics, selected_features, feature_metadata, rejected_features, leaderboard)

    print("Training complete")
    print(f"Search method: {args.search_method}")
    print(f"Search evaluations: {len(leaderboard)}")
    print(f"Selected features ({len(selected_features)}): {', '.join(selected_features)}")
    print(f"Validation ROC-AUC: {metrics['roc_auc']}")
    print(f"Validation Gini: {metrics['gini']}")
    print(f"Threshold: {metrics['classification_threshold']}")
    print(f"Artifacts: {ARTIFACT_DIR / 'default_xgboost_artifacts.joblib'}")
    print(f"Report: {REPORT_DIR / 'model_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
