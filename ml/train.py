from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    make_scorer,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.dataset import (
    CLASSES,
    build_dataset,
    get_categorical_columns,
    get_numeric_columns,
)
from dagster_project.assets.gold import GOLD_CONFIG

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = get_numeric_columns(X)
    categorical_cols = get_categorical_columns(X)

    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value=0)),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


def build_pipelines(X: pd.DataFrame) -> dict[str, Pipeline]:
    preprocessor = build_preprocessor(X)
    return {
        "logreg": Pipeline(
            [
                ("prep", preprocessor),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
            ]
        ),
        "rf": Pipeline(
            [
                ("prep", preprocessor),
                # Để n_jobs=None (hoặc 1) để tránh xung đột luồng lồng nhau với RandomizedSearchCV
                ("clf", RandomForestClassifier(class_weight="balanced", random_state=42)),
            ]
        ),
    }


RF_PARAM_DIST = {
    "clf__n_estimators": [100, 150],
    "clf__max_depth": [10, 15, None],
    "clf__min_samples_split": [2, 5],
    "clf__min_samples_leaf": [1, 2],
}


def tune_rf(pipeline: Pipeline, X_train, y_train, groups):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=RF_PARAM_DIST,
        n_iter=3,
        scoring="f1_macro",
        cv=cv,
        refit=True,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train, groups=groups)
    return search.best_estimator_, search.best_params_


def get_cv_scorers():
    return {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
        "macro_recall": "recall_macro",
        "recall_withdrawn": make_scorer(recall_score, labels=["Withdrawn"], average="macro", zero_division=0),
        "recall_fail": make_scorer(recall_score, labels=["Fail"], average="macro", zero_division=0),
    }


def evaluate_model_cv(pipeline: Pipeline, X_train, y_train, groups) -> dict:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        groups=groups,
        cv=cv,
        scoring=get_cv_scorers(),
        n_jobs=-1,
        return_train_score=False,
    )
    df_metrics = pd.DataFrame(scores).rename(columns=lambda c: c.replace("test_", ""))
    df_metrics = df_metrics.drop(columns=["fit_time", "score_time"], errors="ignore")

    return {
        "per_fold": df_metrics,
        "mean": df_metrics.mean().to_dict(),
        "std": df_metrics.std().to_dict(),
    }


def compare_models(cv_results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, result in cv_results.items():
        rows.append({"model": name, **result["mean"]})
    return pd.DataFrame(rows).set_index("model").round(4)


def evaluate_on_test(final_model: Pipeline, X_test, y_test) -> dict:
    # y_test, y_pred đều là string ("Distinction"/"Pass"/"Fail"/"Withdrawn") —
    # KHÔNG cần LabelEncoder, LogReg/RandomForest nhận nhãn string trực tiếp.
    y_pred = final_model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=CLASSES)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_withdrawn": recall_score(y_test, y_pred, labels=["Withdrawn"], average="macro", zero_division=0),
        "recall_fail": recall_score(y_test, y_pred, labels=["Fail"], average="macro", zero_division=0),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def compute_feature_importance(final_model: Pipeline) -> pd.DataFrame:
    preprocessor = final_model.named_steps["prep"]
    clf = final_model.named_steps["clf"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        # Đối với LogReg multiclass: lấy độ lớn trung bình của hệ số tuyệt đối
        importances = np.mean(np.abs(clf.coef_), axis=0)
    else:
        importances = np.zeros(len(feature_names))

    df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df_imp.sort_values(by="importance", ascending=False).reset_index(drop=True)


def save_artifacts(
    cutoff_name: str,
    best_model_name: str,
    final_model: Pipeline,
    best_params: dict,
    comparison_table: pd.DataFrame,
    test_metrics: dict,
    importance_df: pd.DataFrame,
    cutoff_day: int,
    X_train: pd.DataFrame,
):
    out_dir = os.path.join(MODELS_DIR, cutoff_name)
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump(final_model, os.path.join(out_dir, "best_model.joblib"))
    importance_df.to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)

    numeric_cols = get_numeric_columns(X_train)
    categorical_cols = get_categorical_columns(X_train)

    metadata = {
        "model": best_model_name,
        "cutoff_name": cutoff_name,
        "cutoff_day": cutoff_day,
        "target": "final_result",
        "classes": CLASSES,
        "feature_columns": {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "all": numeric_cols + categorical_cols,
        },
        "best_parameters": best_params,
        "cv_comparison_all_models": comparison_table.to_dict(orient="index"),
        "test_metrics": test_metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)

    print(f"Đã lưu artifacts vào {out_dir}")


def train_for_cutoff(cutoff_name: str, cutoff_day: int):
    print(f"\n{'=' * 60}\nHUẤN LUYỆN CHO {cutoff_name} (ngày {cutoff_day})\n{'=' * 60}")

    # y_train/y_test giữ NGUYÊN dạng string ("Distinction"/"Pass"/"Fail"/
    # "Withdrawn") — KHÔNG dùng LabelEncoder. LogisticRegression/RandomForest
    # nhận nhãn string trực tiếp; LabelEncoder chỉ cần thiết khi còn dùng
    # XGBoost (đã bỏ theo góp ý giảng viên) — giữ lại sẽ gây lệch mapping
    # giữa (a) thứ tự alphabet mà LabelEncoder tự chọn và (b) thứ tự trong
    # CLASSES mà predict.py giả định, dẫn tới lấy sai cột xác suất Fail/Withdrawn.
    X_train, y_train, X_test, y_test, meta_train, _ = build_dataset(cutoff_name)
    groups_train = meta_train["id_student"]

    pipelines = build_pipelines(X_train)
    tuned = {}

    print("-- Fit Baseline Logistic Regression --")
    tuned["logreg"] = {
        "estimator": clone(pipelines["logreg"]).fit(X_train, y_train),
        "best_params": {},
    }

    print("-- Tuning Random Forest --")
    rf_estimator, rf_params = tune_rf(pipelines["rf"], X_train, y_train, groups_train)
    tuned["rf"] = {"estimator": rf_estimator, "best_params": rf_params}

    cv_results = {}
    for name, info in tuned.items():
        print(f"-- Đánh giá CV cho {name} --")
        cv_results[name] = evaluate_model_cv(info["estimator"], X_train, y_train, groups_train)

    comparison_table = compare_models(cv_results)
    print("\n=== BẢNG SO SÁNH 2 MODEL (CV) ===")
    print(comparison_table)

    # Chọn model có Macro F1 cao hơn
    best_name = comparison_table["macro_f1"].idxmax()
    final_model = tuned[best_name]["estimator"]
    best_params = tuned[best_name]["best_params"]
    print(f"\n>>> Model được chọn: {best_name}")

    test_metrics = evaluate_on_test(final_model, X_test, y_test)
    print("\n=== ĐÁNH GIÁ CUỐI TRÊN TEST ===")
    for k in ["accuracy", "macro_f1", "recall_withdrawn", "recall_fail"]:
        print(f"{k}: {test_metrics[k]:.4f}")

    importance_df = compute_feature_importance(final_model)
    save_artifacts(
        cutoff_name,
        best_name,
        final_model,
        best_params,
        comparison_table,
        test_metrics,
        importance_df,
        cutoff_day,
        X_train,
    )

    return final_model, comparison_table, test_metrics


def train_all():
    results = {}
    for cutoff_name, cfg in GOLD_CONFIG.items():
        results[cutoff_name] = train_for_cutoff(cutoff_name, cfg["cutoff_day"])
    return results


if __name__ == "__main__":
    train_all()