import pandas as pd
from dagster import (
    AssetCheckExecutionContext,
    AssetCheckResult,
    AssetExecutionContext,
    MaterializeResult,
    asset,
    asset_check,
)

from dagster_project.assets.gold import GOLD_CONFIG
from ml.predict import predict_all, upsert_predictions
from ml.train import train_all


@asset(
    name="ml_train_models",
    deps=[f"{cutoff_name}" for cutoff_name in GOLD_CONFIG],
    group_name="machine_learning",
    description="Huấn luyện và so sánh 2 model (Logistic Regression & Random Forest) cho từng cutoff",
)
def ml_train_models(context: AssetExecutionContext) -> MaterializeResult:
    results = train_all()

    summary = {}
    for cutoff_name, (final_model, comparison_table, test_metrics) in results.items():
        summary[cutoff_name] = {
            "test_macro_f1": round(float(test_metrics["macro_f1"]), 4),
            "test_recall_withdrawn": round(float(test_metrics["recall_withdrawn"]), 4),
            "test_recall_fail": round(float(test_metrics["recall_fail"]), 4),
        }
        context.log.info(f"[{cutoff_name}] test metrics: {summary[cutoff_name]}")

    return MaterializeResult(
        metadata={f"{k}_metrics": str(v) for k, v in summary.items()}
    )


@asset(
    name="ml_predictions",
    deps=["ml_train_models"],
    group_name="machine_learning",
    description="Dự đoán final_result + at_risk_probability cho tất cả sinh viên ở 4 cutoff, UPSERT vào Postgres ml_results.fact_student_prediction",
)
def ml_predictions(context: AssetExecutionContext) -> MaterializeResult:
    df_pred = predict_all()
    n_rows = upsert_predictions(df_pred)

    context.log.info(f"Đã UPSERT {n_rows} dòng dự đoán vào ml_results.fact_student_prediction")
    return MaterializeResult(
        metadata={
            "n_rows": n_rows,
            "risk_level_distribution": str(df_pred["risk_level"].value_counts().to_dict()),
        }
    )


@asset_check(asset=ml_predictions, description="Kiểm tra tính hợp lệ của kết quả dự đoán")
def check_prediction_probability_range(context: AssetCheckExecutionContext) -> AssetCheckResult:
    df_pred = predict_all()

    prob_cols = ["withdrawn_probability", "fail_probability", "at_risk_probability"]
    prob_in_range = bool(
        ((df_pred[prob_cols] >= 0) & (df_pred[prob_cols] <= 1)).all(axis=None)
    )

    at_risk_consistent = bool(
        (
            df_pred["at_risk_probability"]
            >= df_pred[["withdrawn_probability", "fail_probability"]].max(axis=1) - 1e-9
        ).all()
    )

    valid_risk_level = bool(df_pred["risk_level"].isin(["Low", "Medium", "High"]).all())

    no_duplicate_key = bool(
        not df_pred.duplicated(
            subset=["id_student", "code_module", "code_presentation", "cutoff_week"]
        ).any()
    )

    passed = prob_in_range and at_risk_consistent and valid_risk_level and no_duplicate_key

    return AssetCheckResult(
        passed=passed,
        metadata={
            "prob_in_range_0_1": prob_in_range,
            "at_risk_equals_max_withdrawn_fail": at_risk_consistent,
            "valid_risk_level_values": valid_risk_level,
            "no_duplicate_key": no_duplicate_key,
            "n_rows_checked": int(len(df_pred)),
        },
    )