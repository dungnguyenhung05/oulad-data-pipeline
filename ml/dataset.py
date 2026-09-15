from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from dagster_project.assets.gold import get_duckdb_connection, parquet_path

GOLD_BUCKET = "oulad-gold"

ID_COLS = ["id_student", "code_module", "code_presentation"]
TARGET_COL = "final_result"
CLASSES = ["Distinction", "Pass", "Fail", "Withdrawn"]

# Loại bỏ ID, Target, last_active_day (chống rò rỉ dữ liệu)
# và gender (kiểm định Cramér's V = 0.02 cho thấy không có liên kết với nhãn)
DROP_COLS = ID_COLS + [
    TARGET_COL,
    "last_active_day",
    "gender",
    "active_site",          # tên Gold là active_site, không phải avg_active_sites
    "num_late_submissions", # tương đương khái niệm late_rate ở mart_dashboard
]


def load_gold(cutoff_name: str) -> pd.DataFrame:
    con = get_duckdb_connection()
    try:
        path = parquet_path(GOLD_BUCKET, cutoff_name)
        return con.execute(f"SELECT * FROM read_parquet('{path}')").df()
    finally:
        con.close()


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df[df[TARGET_COL].notna()].reset_index(drop=True)

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    train_idx, test_idx = next(
        cv.split(df, y=df[TARGET_COL], groups=df["id_student"])
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    overlap = set(train_df["id_student"]) & set(test_df["id_student"])
    assert not overlap, f"Phát hiện {len(overlap)} id_student cùng tồn tại ở Train/Test"

    return train_df, test_df


def get_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    y = df[TARGET_COL]
    return X, y


def get_features_for_predict(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in DROP_COLS if c in df.columns])


def get_categorical_columns(X: pd.DataFrame) -> list[str]:
    return X.select_dtypes(include=["object", "category"]).columns.tolist()


def get_numeric_columns(X: pd.DataFrame) -> list[str]:
    return X.select_dtypes(include=["number"]).columns.tolist()


def build_dataset(cutoff_name: str):
    df = load_gold(cutoff_name)
    train_df, test_df = split_train_test(df)

    X_train, y_train = get_features_target(train_df)
    X_test, y_test = get_features_target(test_df)

    meta_train = train_df[ID_COLS].copy().reset_index(drop=True)
    meta_test = test_df[ID_COLS].copy().reset_index(drop=True)

    return (
        X_train.reset_index(drop=True),
        y_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_test.reset_index(drop=True),
        meta_train,
        meta_test
    )