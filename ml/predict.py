from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from dagster_project.assets.gold import GOLD_CONFIG
from sqlalchemy import create_engine, text

from ml.dataset import ID_COLS, load_gold, get_features_for_predict

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
RISK_THRESHOLDS = {"high": 0.6, "medium": 0.3}


def load_model_bundle(cutoff_name: str):
    model_dir = os.path.join(MODELS_DIR, cutoff_name)
    model = joblib.load(os.path.join(model_dir, "best_model.joblib"))
    with open(os.path.join(model_dir, "metadata.json"), encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata


def extract_risk_probabilities(model, X: pd.DataFrame) -> pd.DataFrame:

    proba = model.predict_proba(X)
    classes_ = list(model.classes_)

    withdrawn_probability = proba[:, classes_.index("Withdrawn")]
    fail_probability = proba[:, classes_.index("Fail")]
    at_risk_probability = np.maximum(withdrawn_probability, fail_probability)

    return pd.DataFrame(
        {
            "withdrawn_probability": withdrawn_probability,
            "fail_probability": fail_probability,
            "at_risk_probability": at_risk_probability,
        }
    )


def bucket_risk_level(at_risk_probability: pd.Series) -> pd.Series:
    return pd.cut(
        at_risk_probability,
        bins=[-0.01, RISK_THRESHOLDS["medium"], RISK_THRESHOLDS["high"], 1.0],
        labels=["Low", "Medium", "High"],
    )


def predict_for_cutoff(cutoff_name: str) -> pd.DataFrame:
    model, metadata = load_model_bundle(cutoff_name)

    df = load_gold(cutoff_name)
    X = get_features_for_predict(df)
    meta = df[ID_COLS].reset_index(drop=True)
    X = X.reset_index(drop=True)

    # model.predict() trả thẳng string, không cần map lại từ số nữa.
    predicted_class = model.predict(X)

    risk_df = extract_risk_probabilities(model, X)
    risk_level = bucket_risk_level(risk_df["at_risk_probability"])

    result = pd.concat([meta, risk_df], axis=1)
    result["predicted_final_result"] = predicted_class
    result["risk_level"] = risk_level.astype(str)
    result["cutoff_week"] = metadata["cutoff_day"] // 7
    result["cutoff_name"] = cutoff_name
    result["predicted_at"] = datetime.now(timezone.utc)

    return result


def predict_all() -> pd.DataFrame:
    frames = [predict_for_cutoff(cutoff_name) for cutoff_name in GOLD_CONFIG]
    return pd.concat(frames, ignore_index=True)


def get_student_profile(cutoff_name: str, id_student: int, code_module: str, code_presentation: str) -> pd.DataFrame:
    """
    Trích xuất đóng góp đặc trưng (Feature Impact) của sinh viên dựa trên
    Feature Importance tổng hợp từ mô hình (thay thế SHAP).
    """
    pipeline, metadata = load_model_bundle(cutoff_name)
    df = load_gold(cutoff_name)

    row = df[
        (df["id_student"] == id_student)
        & (df["code_module"] == code_module)
        & (df["code_presentation"] == code_presentation)
    ]
    if row.empty:
        raise ValueError("Không tìm thấy sinh viên/môn/kỳ học này ở cutoff đã chọn.")

    X_row = get_features_for_predict(row)
    preprocessor = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]

    X_transformed = preprocessor.transform(X_row)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    feature_names = list(preprocessor.get_feature_names_out())

    if hasattr(clf, "feature_importances_"):
        base_impact = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        base_impact = np.mean(np.abs(clf.coef_), axis=0)
    else:
        base_impact = np.zeros(len(feature_names))

    individual_impact = X_transformed[0] * base_impact

    df_profile = pd.DataFrame({
        "feature": [f.replace("num__", "").replace("cat__", "") for f in feature_names],
        "feature_impact": individual_impact,
        "base_importance": base_impact,
    }).sort_values(by="feature_impact", ascending=False).reset_index(drop=True)

    return df_profile


def get_postgres_engine():
    # Hỗ trợ cả POSTGRES_HOST lẫn POSTGRES_APP_HOST
    host = os.environ.get("POSTGRES_APP_HOST", "postgres")
    port = os.environ.get("POSTGRES_APP_PORT", "5433")
    user = os.environ.get("POSTGRES_APP_USER", "postgres")
    password = os.environ.get("POSTGRES_APP_PASSWORD", "")
    db = os.environ.get("POSTGRES_APP_DB", "oulad_dwh")

    return create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")


CREATE_SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS ml_results;
"""

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ml_results.fact_student_prediction (
    id_student              BIGINT NOT NULL,
    code_module             TEXT NOT NULL,
    code_presentation       TEXT NOT NULL,
    cutoff_week             INTEGER NOT NULL,
    cutoff_name             TEXT NOT NULL,
    predicted_final_result  TEXT NOT NULL,
    withdrawn_probability   DOUBLE PRECISION NOT NULL,
    fail_probability        DOUBLE PRECISION NOT NULL,
    at_risk_probability     DOUBLE PRECISION NOT NULL,
    risk_level              TEXT NOT NULL,
    predicted_at            TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (id_student, code_module, code_presentation, cutoff_week)
);
"""

UPSERT_SQL = """
INSERT INTO ml_results.fact_student_prediction (
    id_student, code_module, code_presentation, cutoff_week, cutoff_name,
    predicted_final_result, withdrawn_probability, fail_probability,
    at_risk_probability, risk_level, predicted_at
) VALUES (
    :id_student, :code_module, :code_presentation, :cutoff_week, :cutoff_name,
    :predicted_final_result, :withdrawn_probability, :fail_probability,
    :at_risk_probability, :risk_level, :predicted_at
)
ON CONFLICT (id_student, code_module, code_presentation, cutoff_week)
DO UPDATE SET
    cutoff_name             = EXCLUDED.cutoff_name,
    predicted_final_result  = EXCLUDED.predicted_final_result,
    withdrawn_probability   = EXCLUDED.withdrawn_probability,
    fail_probability        = EXCLUDED.fail_probability,
    at_risk_probability     = EXCLUDED.at_risk_probability,
    risk_level              = EXCLUDED.risk_level,
    predicted_at            = EXCLUDED.predicted_at;
"""


def upsert_predictions(df: pd.DataFrame) -> int:
    engine = get_postgres_engine()
    records = df.to_dict(orient="records")

    with engine.begin() as conn:
        # 1. Tạo Schema trước
        conn.execute(text(CREATE_SCHEMA_SQL))
        # 2. Tạo Bảng riêng rẽ
        conn.execute(text(CREATE_TABLE_SQL))
        # 3. Thực thi batch upsert dữ liệu
        if records:
            conn.execute(text(UPSERT_SQL), records)

    return len(records)


if __name__ == "__main__":
    print("⏳ Đang chạy dự đoán cho tất cả các mốc Cutoff...")
    df_all_preds = predict_all()

    print("\n--- Phân bố Risk Level tổng hợp ---")
    print(df_all_preds["risk_level"].value_counts())

    print("\n⏳ Đang UPSERT dữ liệu vào PostgreSQL (ml_results.fact_student_prediction)...")
    total_rows = upsert_predictions(df_all_preds)
    print(f"✅ Đã UPSERT thành công {total_rows:,} dòng vào Database!")