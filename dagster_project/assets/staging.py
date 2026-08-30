import pandas as pd
from dagster import asset, AssetExecutionContext

from dagster_project.resources import get_postgres_engine


def write_staging_table(df: pd.DataFrame, table_name: str, schema: str = "staging_ml") -> None:
    engine = get_postgres_engine()
    with engine.connect() as conn:
        exists = conn.exec_driver_sql(
            f"SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            f"WHERE table_schema='{schema}' AND table_name='{table_name}')"
        ).scalar()

    if exists:
        with engine.begin() as conn:
            conn.exec_driver_sql(f"TRUNCATE TABLE {schema}.{table_name}")
        df.to_sql(table_name, engine, schema=schema, if_exists="append", index=False, method="multi", chunksize=5000)
    else:
        df.to_sql(table_name, engine, schema=schema, if_exists="replace", index=False, method="multi", chunksize=5000)


def build_staging_asset(
    context: AssetExecutionContext,
    df: pd.DataFrame,
    table_name: str,
) -> None:
    write_staging_table(df, table_name)
    context.add_output_metadata({
        "rows_loaded": len(df),
    })



@asset(group_name="staging_ml")
def stg_gold_features_cutoff2(context: AssetExecutionContext, gold_features_cutoff2: pd.DataFrame) -> None:
    build_staging_asset(context, gold_features_cutoff2, "stg_gold_features_cutoff2")


@asset(group_name="staging_ml")
def stg_gold_features_cutoff4(context: AssetExecutionContext, gold_features_cutoff4: pd.DataFrame) -> None:
    build_staging_asset(context, gold_features_cutoff4, "stg_gold_features_cutoff4")


@asset(group_name="staging_ml")
def stg_gold_features_cutoff8(context: AssetExecutionContext, gold_features_cutoff8: pd.DataFrame) -> None:
    build_staging_asset(context, gold_features_cutoff8, "stg_gold_features_cutoff8")


@asset(group_name="staging_ml")
def stg_gold_features_cutoff12(context: AssetExecutionContext, gold_features_cutoff12: pd.DataFrame) -> None:
    build_staging_asset(context, gold_features_cutoff12, "stg_gold_features_cutoff12")

