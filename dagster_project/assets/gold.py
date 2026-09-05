import pandas as pd
from dagster import asset, AssetExecutionContext, MetadataValue, AssetKey
import duckdb
import os


GOLD_CONFIG = {
    "gold_features_cutoff2": {"cutoff_day": 14},
    "gold_features_cutoff4": {"cutoff_day": 28},
    "gold_features_cutoff8": {"cutoff_day": 56},
    "gold_features_cutoff12": {"cutoff_day": 84},
}
BRONZE_BUCKET = "oulad-bronze"
SILVER_BUCKET = "oulad-silver"

# Khoa merge bang
KEYS = ["id_student", "code_module", "code_presentation"]


def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("LOAD httpfs;")  # khong install nua, install tu dau o dockerfile

    endpoint = os.environ["MINIO_ENDPOINT_URL"].replace("http://", "").replace("https://", "")
    con.execute(f"""
        SET s3_endpoint='{endpoint}';
        SET s3_access_key_id='{os.environ["MINIO_ROOT_USER"]}';
        SET s3_secret_access_key='{os.environ["MINIO_ROOT_PASSWORD"]}';
        SET s3_url_style='path';
        SET s3_use_ssl=false;
    """)
    return con

# tao ten cho file parquet
def _parquet_path(bucket: str, asset_name: str) -> str:
    return f"s3://{bucket}/{asset_name}/{asset_name}.parquet"



# Tao base cho bang gold
def get_base_population(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    path = _parquet_path(BRONZE_BUCKET, "bronze_student_registration")
    query = f"""
        SELECT id_student, code_module, code_presentation, date_registration
        FROM read_parquet('{path}')
    """
    return con.execute(query).df()


# Feature cua bronze_student_info (Demographic) lay data tu bang bronze
def get_demographic_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    path = _parquet_path(BRONZE_BUCKET, "bronze_student_info")
    query = f"""
        SELECT id_student, code_module, code_presentation,
               age_band, imd_band, highest_education, region,
               num_of_prev_attempts, studied_credits, disability
        FROM read_parquet('{path}')
    """
    return con.execute(query).df()



# feature cua silver_student_vle_enriched (So luot click)
def get_engagement_features(con: duckdb.DuckDBPyConnection, cutoff_day: int) -> pd.DataFrame:
    path = _parquet_path(SILVER_BUCKET, "silver_student_vle_enriched")
    query = f"""
        SELECT
            id_student, code_module, code_presentation,
            SUM(sum_click) AS total_click,
            COUNT(DISTINCT date) AS active_days,
            COUNT(DISTINCT id_site) AS active_site,
            COUNT(DISTINCT activity_type) AS activity_types,
            MAX(date) AS last_active_day
        FROM read_parquet('{path}')
        WHERE date <= {cutoff_day}
        GROUP BY id_student, code_module, code_presentation
    """
    result = con.execute(query).df()
    result["avg_click_per_day"] = result["total_click"] / result["active_days"]
    result["days_since_last_activity"] = cutoff_day - result["last_active_day"]
    return result


# Feature cua silver_student_assessment_enriched (So bai nop)
def get_assignment_catalog(con: duckdb.DuckDBPyConnection, cutoff_day: int) -> pd.DataFrame:
    path = _parquet_path(BRONZE_BUCKET, "bronze_assessments")
    query = f"""
        SELECT code_module, code_presentation,
               COUNT(DISTINCT id_assessment) AS num_assigned
        FROM read_parquet('{path}')
        WHERE date <= {cutoff_day}
        GROUP BY code_module, code_presentation
    """
    return con.execute(query).df()


def get_assessment_features(con: duckdb.DuckDBPyConnection, cutoff_day: int) -> pd.DataFrame:
    path = _parquet_path(SILVER_BUCKET, "silver_student_assessment_enriched")
    query = f"""
        WITH submitted AS (
            SELECT * FROM read_parquet('{path}')
            WHERE date_submitted <= {cutoff_day}
        ),
        late AS (
            SELECT * FROM submitted WHERE date_submitted > date
        )
        SELECT
            s.id_student, s.code_module, s.code_presentation,
            COUNT(s.date_submitted) AS num_submission,
            AVG(s.score) AS avg_score,
            COUNT(DISTINCT l.id_assessment) AS num_late_submissions
        FROM submitted s
        LEFT JOIN late l
            ON s.id_student = l.id_student
           AND s.code_module = l.code_module
           AND s.code_presentation = l.code_presentation
           AND s.id_assessment = l.id_assessment
        GROUP BY s.id_student, s.code_module, s.code_presentation
    """
    return con.execute(query).df()

# Xu ly 0/ NaN
def fill_missing_gold(base: pd.DataFrame, cutoff_day: int) -> pd.DataFrame:
    cols_fill_zero = [
        "total_click", "active_days", "active_site", "activity_types",
        "num_assigned", "num_submission", "num_late_submissions",
        "num_of_prev_attempts", "studied_credits",
    ]
    cols_fill_unknown = ["age_band", "imd_band", "highest_education", "region", "disability"]

    base[cols_fill_zero] = base[cols_fill_zero].fillna(0)
    base[cols_fill_unknown] = base[cols_fill_unknown].fillna("Unknown")
    # Them feature sau khi fillna 0
    base["submission_rate"] = base["num_submission"] / base["num_assigned"]
    # neu NaN thi so ngay im lang toi da la cutoff_day (khong dien 0)
    base["days_since_last_activity"] = base["days_since_last_activity"].fillna(cutoff_day)

    return base



# Tao bang du lieu gold (gop base voi cac feature)
def build_gold_features(context: AssetExecutionContext, asset_name: str) -> pd.DataFrame:
    cutoff_day = GOLD_CONFIG[asset_name]["cutoff_day"]
    con = get_duckdb_connection()

    try:
        base = get_base_population(con)
        engagement = get_engagement_features(con, cutoff_day)
        assignment_catalog = get_assignment_catalog(con, cutoff_day)
        assessment = get_assessment_features(con, cutoff_day)
        demographic = get_demographic_features(con)
    finally:
        con.close()

    result = (
        base
        .merge(engagement, on=KEYS, how="left")
        .merge(assignment_catalog, on=["code_module", "code_presentation"], how="left")
        .merge(assessment, on=KEYS, how="left")
        .merge(demographic, on=KEYS, how="left")
    )
    result = fill_missing_gold(result, cutoff_day)

    preview_md = result.head(5).to_markdown(index=False)
    context.add_output_metadata({
        "cutoff_day": MetadataValue.int(cutoff_day),
        "num_columns": MetadataValue.int(len(result.columns)),
        "missing_values": MetadataValue.int(int(result.isna().sum().sum())),
        "data_preview": MetadataValue.md(preview_md),
    })

    return result


_gold_deps = [
    AssetKey("bronze_student_registration"),
    AssetKey("bronze_student_info"),
    AssetKey("bronze_assessments"),
    AssetKey("silver_student_vle_enriched"),
    AssetKey("silver_student_assessment_enriched"),
]


@asset(group_name="gold", io_manager_key="gold_io_manager", deps=_gold_deps)
def gold_features_cutoff2(context: AssetExecutionContext) -> pd.DataFrame:
    return build_gold_features(context, "gold_features_cutoff2")


@asset(group_name="gold", io_manager_key="gold_io_manager", deps=_gold_deps)
def gold_features_cutoff4(context: AssetExecutionContext) -> pd.DataFrame:
    return build_gold_features(context, "gold_features_cutoff4")


@asset(group_name="gold", io_manager_key="gold_io_manager", deps=_gold_deps)
def gold_features_cutoff8(context: AssetExecutionContext) -> pd.DataFrame:
    return build_gold_features(context, "gold_features_cutoff8")


@asset(group_name="gold", io_manager_key="gold_io_manager", deps=_gold_deps)
def gold_features_cutoff12(context: AssetExecutionContext) -> pd.DataFrame:
    return build_gold_features(context, "gold_features_cutoff12")



