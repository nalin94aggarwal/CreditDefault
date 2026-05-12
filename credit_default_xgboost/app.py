from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb


PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = PROJECT_DIR / "artifacts" / "default_xgboost_artifacts.joblib"
REPORT_DIR = PROJECT_DIR / "reports"


@st.cache_resource
def load_artifacts(artifact_mtime_ns: int) -> Dict[str, object]:
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts were not found. Run `python train_model.py` before launching the app."
        )
    return joblib.load(ARTIFACT_PATH)


def format_value(value: float, input_type: str) -> str:
    if input_type == "currency":
        return f"${value:,.0f}"
    if input_type == "percent":
        return f"{value:.3f}"
    if input_type in {"integer", "binary"}:
        return f"{int(round(value))}"
    return f"{value:,.3f}"


def typicality(value: float, stats: Dict[str, float]) -> str:
    if value >= stats["p75"]:
        return "higher than typical"
    if value <= stats["p25"]:
        return "lower than typical"
    return "near the typical range"


def number_input_for_feature(
    feature: str,
    metadata: Dict[str, object],
    key_prefix: str,
) -> float:
    stats = metadata["stats"]
    input_type = str(metadata["input_type"])
    label = str(metadata["label"])
    help_text = str(metadata["rationale"])
    default = float(stats["median"])

    if input_type == "binary":
        default_index = int(round(default))
        value = st.selectbox(
            label,
            options=[0, 1],
            index=default_index if default_index in [0, 1] else 0,
            format_func=lambda x: "Yes" if x == 1 else "No",
            help=help_text,
            key=f"{key_prefix}_{feature}",
        )
        return float(value)

    if input_type == "integer":
        return float(
            st.number_input(
                label,
                value=int(round(default)),
                step=1,
                help=help_text,
                key=f"{key_prefix}_{feature}",
            )
        )

    if input_type == "currency":
        return float(
            st.number_input(
                label,
                value=float(round(default, 2)),
                step=100.0,
                format="%.2f",
                help=help_text,
                key=f"{key_prefix}_{feature}",
            )
        )

    return float(
        st.number_input(
            label,
            value=float(round(default, 4)),
            step=0.01,
            format="%.4f",
            help=help_text,
            key=f"{key_prefix}_{feature}",
        )
    )


def prepare_input(values: Dict[str, float], selected_features: List[str], imputer) -> pd.DataFrame:
    raw = pd.DataFrame([values], columns=selected_features)
    transformed = imputer.transform(raw[selected_features])
    return pd.DataFrame(transformed, columns=selected_features)


def predict_probability(
    model,
    imputer,
    selected_features: List[str],
    values: Dict[str, float],
) -> float:
    input_df = prepare_input(values, selected_features, imputer)
    return float(model.predict_proba(input_df[selected_features])[:, 1][0])


def feature_sensitivity_table(
    model,
    imputer,
    selected_features: List[str],
    feature_metadata: Dict[str, Dict[str, object]],
    current_values: Dict[str, float],
    current_probability: float,
) -> pd.DataFrame:
    rows = []
    for feature in selected_features:
        metadata = feature_metadata[feature]
        stats = metadata["stats"]
        input_type = str(metadata["input_type"])

        if input_type == "binary":
            low_value = 0.0
            high_value = 1.0
        elif input_type == "integer":
            low_value = float(round(stats["p05"]))
            high_value = float(round(stats["p95"]))
        else:
            low_value = float(stats["p05"])
            high_value = float(stats["p95"])

        low_values = dict(current_values)
        high_values = dict(current_values)
        low_values[feature] = low_value
        high_values[feature] = high_value

        low_probability = predict_probability(model, imputer, selected_features, low_values)
        high_probability = predict_probability(model, imputer, selected_features, high_values)
        max_change = max(
            abs(low_probability - current_probability),
            abs(high_probability - current_probability),
        )

        rows.append(
            {
                "Feature": metadata["label"],
                "Current value": format_value(float(current_values[feature]), input_type),
                "Low test value": format_value(low_value, input_type),
                "Probability at low": f"{low_probability:.3%}",
                "High test value": format_value(high_value, input_type),
                "Probability at high": f"{high_probability:.3%}",
                "Max change from current": f"{max_change:.3%}",
                "Risk direction": (
                    "higher value raises risk"
                    if high_probability > low_probability
                    else "higher value lowers risk"
                    if high_probability < low_probability
                    else "flat in this range"
                ),
            }
        )

    sensitivity = pd.DataFrame(rows)
    sensitivity["_sort_value"] = sensitivity["Max change from current"].str.rstrip("%").astype(float)
    return sensitivity.sort_values("_sort_value", ascending=False).drop(columns=["_sort_value"])


def reason_codes(
    model,
    input_df: pd.DataFrame,
    selected_features: List[str],
    feature_metadata: Dict[str, Dict[str, object]],
    prediction_is_default: bool,
) -> pd.DataFrame:
    booster = model.get_booster()
    dmatrix = xgb.DMatrix(input_df[selected_features], feature_names=selected_features)
    contributions = booster.predict(dmatrix, pred_contribs=True)[0][:-1]

    rows = []
    for feature, contribution in zip(selected_features, contributions):
        metadata = feature_metadata[feature]
        stats = metadata["stats"]
        value = float(input_df.iloc[0][feature])
        rows.append(
            {
                "feature": feature,
                "reason": metadata["label"],
                "value": format_value(value, str(metadata["input_type"])),
                "typicality": typicality(value, stats),
                "model_contribution": float(contribution),
                "direction": "increases default risk" if contribution > 0 else "reduces default risk",
                "plain_english": metadata["rationale"],
            }
        )

    reason_df = pd.DataFrame(rows)
    if prediction_is_default:
        ranked = reason_df.sort_values("model_contribution", ascending=False)
        ranked = ranked[ranked["model_contribution"] > 0]
        if ranked.empty:
            ranked = reason_df.reindex(reason_df["model_contribution"].abs().sort_values(ascending=False).index)
    else:
        ranked = reason_df.sort_values("model_contribution", ascending=True)
        ranked = ranked[ranked["model_contribution"] < 0]
        if ranked.empty:
            ranked = reason_df.reindex(reason_df["model_contribution"].abs().sort_values(ascending=False).index)

    return ranked.head(3).reset_index(drop=True)


def render_metric_card(label: str, value: object) -> None:
    st.metric(label=label, value=value)


def read_report_csv(file_name: str) -> Optional[pd.DataFrame]:
    path = REPORT_DIR / file_name
    if not path.exists():
        return None
    return pd.read_csv(path)


def main() -> None:
    st.set_page_config(
        page_title="Credit Default XGBoost Scorer",
        layout="wide",
    )

    st.title("Credit Default XGBoost Scorer")
    st.caption("Compact explainable model using fewer than 20 low-correlation credit behavior variables.")

    try:
        artifacts = load_artifacts(ARTIFACT_PATH.stat().st_mtime_ns if ARTIFACT_PATH.exists() else 0)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    model = artifacts["model"]
    imputer = artifacts["imputer"]
    selected_features = artifacts["selected_features"]
    feature_metadata = artifacts["feature_metadata"]
    threshold = float(artifacts["threshold"])
    metrics = artifacts["metrics"]

    tab_score, tab_model, tab_validation, tab_pdp = st.tabs(
        ["Prediction", "Model Details", "Validation Distribution", "Partial Dependence"]
    )

    with tab_score:
        st.subheader("Applicant Inputs")
        st.write("Enter the credit behavior and financial capacity inputs below.")

        values: Dict[str, float] = {}
        columns = st.columns(2)
        for index, feature in enumerate(selected_features):
            with columns[index % 2]:
                values[feature] = number_input_for_feature(
                    feature,
                    feature_metadata[feature],
                    key_prefix="score",
                )

        input_df = prepare_input(values, selected_features, imputer)
        probability = float(model.predict_proba(input_df[selected_features])[:, 1][0])
        prediction_is_default = probability >= threshold
        decision = "Default risk flagged" if prediction_is_default else "Default risk not flagged"

        st.divider()
        result_cols = st.columns(4)
        with result_cols[0]:
            render_metric_card("Default probability", f"{probability:.3%}")
        with result_cols[1]:
            render_metric_card("Decision threshold", f"{threshold:.3%}")
        with result_cols[2]:
            render_metric_card("Prediction", decision)
        with result_cols[3]:
            render_metric_card("Model AUC", metrics.get("roc_auc", "n/a"))

        st.subheader("Top 3 Reason Codes")
        reasons = reason_codes(
            model,
            input_df,
            selected_features,
            feature_metadata,
            prediction_is_default,
        )
        st.dataframe(
            reasons[
                [
                    "reason",
                    "value",
                    "typicality",
                    "direction",
                    "plain_english",
                    "model_contribution",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Reason codes are XGBoost contribution estimates for this single prediction. "
            "They explain model behavior, not legal adverse-action decisions or causal effects."
        )

        st.subheader("Input Sensitivity")
        st.write(
            "This table changes one field at a time to its low and high validation-range values. "
            "Tree models can look flat for small edits because the score changes only when a learned split is crossed."
        )
        sensitivity = feature_sensitivity_table(
            model,
            imputer,
            selected_features,
            feature_metadata,
            values,
            probability,
        )
        st.dataframe(sensitivity, use_container_width=True, hide_index=True)

    with tab_model:
        st.subheader("Validation Summary")
        metric_cols = st.columns(4)
        with metric_cols[0]:
            render_metric_card("ROC-AUC", metrics.get("roc_auc", "n/a"))
        with metric_cols[1]:
            render_metric_card("Gini", metrics.get("gini", "n/a"))
        with metric_cols[2]:
            render_metric_card("PR-AUC", metrics.get("average_precision_pr_auc", "n/a"))
        with metric_cols[3]:
            render_metric_card("KS", metrics.get("ks_statistic", "n/a"))

        st.write("Primary measurement method: **ROC-AUC / Gini**.")
        st.write(
            "ROC-AUC is used as the primary measure because default modeling is mainly a "
            "ranking problem, and Gini is the common credit-risk interpretation of AUC."
        )

        st.subheader("Hyperparameter Search")
        st.write(f"Search method: **{metrics.get('hyperparameter_search_method', 'n/a')}**")
        st.write(f"Objective: **{metrics.get('hyperparameter_search_objective', 'n/a')}**")
        st.write(f"Evaluated configurations: **{metrics.get('hyperparameter_search_evaluations', 'n/a')}**")
        best_hyperparameters = metrics.get("best_hyperparameters")
        if best_hyperparameters:
            st.json(best_hyperparameters)

        st.subheader("Selected Model Variables")
        feature_rows = []
        for feature in selected_features:
            item = feature_metadata[feature]
            feature_rows.append(
                {
                    "Feature": item["label"],
                    "Technical name": feature,
                    "Group": item["group"],
                    "Why it is explainable": item["rationale"],
                }
            )
        st.dataframe(pd.DataFrame(feature_rows), use_container_width=True, hide_index=True)

        report_path = REPORT_DIR / "model_report.md"
        if report_path.exists():
            with st.expander("Full model report"):
                st.markdown(report_path.read_text(encoding="utf-8"))

    with tab_validation:
        st.subheader("Validation Prediction Distribution")
        st.write(
            "This view shows how predicted default probabilities are distributed across "
            "the 30% validation split, separated by actual default outcome."
        )

        distribution_path = REPORT_DIR / "validation_score_distribution.png"
        decile_lift_path = REPORT_DIR / "validation_decile_lift.png"
        decile_capture_path = REPORT_DIR / "validation_decile_capture.png"
        if distribution_path.exists():
            st.image(str(distribution_path), caption="Validation predicted probability distribution")
        else:
            st.info("Run `python train_model.py --data ../credit.csv` to generate the distribution plot.")

        quantiles = read_report_csv("validation_score_quantiles.csv")
        if quantiles is not None:
            st.subheader("Score Quantiles")
            st.dataframe(quantiles, use_container_width=True, hide_index=True)

        deciles = read_report_csv("validation_score_deciles.csv")
        if deciles is not None:
            st.subheader("Risk Decile Lift And Capture")
            st.write(
                "Decile 1 contains the highest-risk validation records. "
                "Capture rate shows the share of all validation defaults captured by each decile or cumulative decile group."
            )
            st.dataframe(deciles, use_container_width=True, hide_index=True)

        if decile_lift_path.exists():
            st.image(str(decile_lift_path), caption="Observed default rate by predicted-risk decile")

        if decile_capture_path.exists():
            st.image(str(decile_capture_path), caption="Cumulative default capture by predicted-risk decile")

        predictions = read_report_csv("validation_predictions.csv")
        if predictions is not None:
            with st.expander("Raw validation predictions"):
                st.dataframe(predictions, use_container_width=True, hide_index=True)

    with tab_pdp:
        st.subheader("Partial Dependence Plots")
        summary_path = REPORT_DIR / "partial_dependence_summary.csv"
        if summary_path.exists():
            summary = pd.read_csv(summary_path)
            st.dataframe(summary, use_container_width=True, hide_index=True)
            for _, row in summary.iterrows():
                image_path = PROJECT_DIR / row["plot_file"]
                if image_path.exists():
                    st.image(str(image_path), caption=row["label"], use_column_width=True)
        else:
            st.info("Partial dependence reports are created by running `python train_model.py`.")


if __name__ == "__main__":
    main()
