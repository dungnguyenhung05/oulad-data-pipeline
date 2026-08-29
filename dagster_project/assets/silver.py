import pandas as pd
import io
from dagster import asset, AssetExecutionContext, MetadataValue



@asset(group_name="silver", io_manager_key="silver_io_manager")
def silver_student_vle_enriched(
        context: AssetExecutionContext,
        bronze_student_vle: pd.DataFrame,
        bronze_vle: pd.DataFrame
) -> pd.DataFrame:

    rows_before = len(bronze_student_vle)
    df_merge = pd.merge(bronze_student_vle, bronze_vle, on=["id_site", "code_module", "code_presentation"], how="left")
    rows_after = len(df_merge)

    context.add_output_metadata({
        "rows_before": MetadataValue.int(rows_before),
        "rows_after": MetadataValue.int(rows_after),
        "rows_removed": MetadataValue.int(rows_before - rows_after)
    })

    return df_merge

@asset(group_name="silver", io_manager_key="silver_io_manager")
def silver_student_assessment_enriched(
        context: AssetExecutionContext,
        bronze_student_assessment: pd.DataFrame,
        bronze_assessments: pd.DataFrame
) -> pd.DataFrame:
    rows_before = len(bronze_student_assessment)

    df_merge = pd.merge(bronze_student_assessment, bronze_assessments, on="id_assessment", how="left")
    rows_after = len(df_merge)
    context.add_output_metadata({
        "rows_before": MetadataValue.int(rows_before),
        "rows_after": MetadataValue.int(rows_after),
        "rows_removed": MetadataValue.int(rows_before - rows_after)
    })

    return df_merge

@asset(group_name="silver", io_manager_key="silver_io_manager")
def silver_student_registration_clean(
        context: AssetExecutionContext,
        bronze_student_registration: pd.DataFrame,
) -> pd.DataFrame:
    rows_before = len(bronze_student_registration)

    bronze_student_registration = bronze_student_registration.drop_duplicates(subset=["id_student", "code_module", "code_presentation"])
    rows_after = len(bronze_student_registration)
    context.add_output_metadata({
        "rows_before": MetadataValue.int(rows_before),
        "rows_after": MetadataValue.int(rows_after),
        "rows_removed": MetadataValue.int(rows_before - rows_after)
    })

    return bronze_student_registration


