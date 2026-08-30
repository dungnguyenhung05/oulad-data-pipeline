import pandas as pd
import io
from dagster import asset, AssetExecutionContext

from dagster_project.resources import get_postgres_engine

# Nap df
def write_staging_table(df: pd.DataFrame, table_name: str, schema: str, use_copy: bool = False) -> None:
    engine = get_postgres_engine()
    with engine.connect() as conn:
        exists = conn.exec_driver_sql(
            f"SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            f"WHERE table_schema='{schema}' AND table_name='{table_name}')"
        ).scalar()

    if not exists:
        # Lan dau: tao bang rong dung schema tu DataFrame (khong ghi du lieu)
        df.head(0).to_sql(table_name, engine, schema=schema, if_exists="replace", index=False)

    with engine.begin() as conn:
        conn.exec_driver_sql(f"TRUNCATE TABLE {schema}.{table_name}")

    if use_copy:
        buffer = io.StringIO()
        df.to_csv(buffer, index=False, header=False)
        buffer.seek(0)
        raw_conn = engine.raw_connection()
        try:
            with raw_conn.cursor() as cursor:
                cursor.copy_expert(f"COPY {schema}.{table_name} FROM STDIN WITH CSV", buffer)
            raw_conn.commit()
        finally:
            raw_conn.close()
    else:
        df.to_sql(table_name, engine, schema=schema, if_exists="append",
                  index=False, method="multi", chunksize=5000)


# Tao asset
def build_staging_asset(
    context: AssetExecutionContext,
    df: pd.DataFrame,
    table_name: str,
    schema: str,
    use_copy: bool = False
) -> None:
    write_staging_table(df, table_name, schema, use_copy)
    context.add_output_metadata({
        "rows_loaded": len(df),
        "schema": schema,
        "table_name": table_name
    })



# 4 asset cho ml
@asset(group_name="staging_ml")
def stg_gold_features_cutoff2(context: AssetExecutionContext, gold_features_cutoff2: pd.DataFrame) -> None:
    build_staging_asset(context, gold_features_cutoff2,
                        "stg_gold_features_cutoff2", "staging_ml")


@asset(group_name="staging_ml")
def stg_gold_features_cutoff4(context: AssetExecutionContext, gold_features_cutoff4: pd.DataFrame) -> None:
    build_staging_asset(context, gold_features_cutoff4,
                        "stg_gold_features_cutoff4", "staging_ml")


@asset(group_name="staging_ml")
def stg_gold_features_cutoff8(context: AssetExecutionContext, gold_features_cutoff8: pd.DataFrame) -> None:
    build_staging_asset(context, gold_features_cutoff8,
                        "stg_gold_features_cutoff8", "staging_ml")


@asset(group_name="staging_ml")
def stg_gold_features_cutoff12(context: AssetExecutionContext, gold_features_cutoff12: pd.DataFrame) -> None:
    build_staging_asset(context, gold_features_cutoff12,
                        "stg_gold_features_cutoff12", "staging_ml")



# 6 asset cho dashboard
@asset(group_name="staging_dashboard")
def stg_silver_student_vle_enriched(
        context: AssetExecutionContext,
        silver_student_vle_enriched: pd.DataFrame
) -> None:
    build_staging_asset(context, silver_student_vle_enriched,
                        "stg_silver_student_vle_enriched", "staging_dashboard",
                        use_copy=True)


@asset(group_name="staging_dashboard")
def stg_silver_student_assessment_enriched(
        context: AssetExecutionContext,
        silver_student_assessment_enriched: pd.DataFrame
) -> None:
    build_staging_asset(context, silver_student_assessment_enriched,
                        "stg_silver_student_assessment_enriched", "staging_dashboard")

@asset(group_name="staging_dashboard")
def stg_bronze_student_registration(
    context: AssetExecutionContext,
    bronze_student_registration: pd.DataFrame,
) -> None:
    build_staging_asset(context, bronze_student_registration,
                        "stg_bronze_student_registration", "staging_dashboard")

@asset(group_name="staging_dashboard")
def stg_bronze_student_info(
    context: AssetExecutionContext,
    bronze_student_info: pd.DataFrame,
) -> None:
    build_staging_asset(context, bronze_student_info,
                        "stg_bronze_student_info", "staging_dashboard")

@asset(group_name="staging_dashboard")
def stg_bronze_courses(
    context: AssetExecutionContext,
    bronze_courses: pd.DataFrame,
) -> None:
    build_staging_asset(context, bronze_courses,
                        "stg_bronze_courses", "staging_dashboard")

@asset(group_name="staging_dashboard")
def stg_bronze_assessments(
    context: AssetExecutionContext,
    bronze_assessments: pd.DataFrame,
) -> None:
    build_staging_asset(context, bronze_assessments,
                        "stg_bronze_assessments", "staging_dashboard")