from dagster import Definitions
from dagster_dbt import DbtCliResource
from dagster_project.assets.dbt_assets import oulad_dbt_assets, oulad_dbt_project

from dagster_project.assets.bronze import (
    bronze_student_info,
    bronze_student_registration,
    bronze_student_assessment,
    bronze_assessments,
    bronze_courses,
    bronze_vle,
    bronze_student_vle,
)


from dagster_project.assets.silver import (
    silver_student_vle_enriched,
    silver_student_assessment_enriched,
)


from dagster_project.assets.gold import (
    gold_features_cutoff2,
    gold_features_cutoff4,
    gold_features_cutoff8,
    gold_features_cutoff12,
    check_cutoff2,
    check_cutoff4,
    check_cutoff8,
    check_cutoff12
)


from dagster_project.assets.staging import (
    # dashboard
    stg_silver_student_assessment_enriched,
    stg_silver_student_vle_enriched,
    stg_bronze_student_registration,
    stg_bronze_assessments,
    stg_bronze_courses,
    stg_bronze_student_info
)

from dagster_project.assets.ml import (
    ml_train_models,
    ml_predictions,
    check_prediction_probability_range,
)

from dagster_project.io_manager import MinioParquetIOManager

from dagster_project.resources import get_minio_resource



minio_resource = get_minio_resource()

defs = Definitions(
    assets=[
        # bronze
        bronze_student_info,
        bronze_student_registration,
        bronze_student_assessment,
        bronze_assessments,
        bronze_courses,
        bronze_vle,
        bronze_student_vle,
        # silver
        silver_student_vle_enriched,
        silver_student_assessment_enriched,
        # gold
        gold_features_cutoff2,
        gold_features_cutoff4,
        gold_features_cutoff8,
        gold_features_cutoff12,
        # staging
        stg_silver_student_assessment_enriched,
        stg_silver_student_vle_enriched,
        stg_bronze_student_registration,
        stg_bronze_assessments,
        stg_bronze_courses,
        stg_bronze_student_info,

        oulad_dbt_assets,

        # machine learning assets
        ml_train_models,
        ml_predictions
    ],
    asset_checks=[
        check_cutoff2,
        check_cutoff4,
        check_cutoff8,
        check_cutoff12,
        check_prediction_probability_range
    ],
    resources={
        's3': minio_resource,
        'bronze_io_manager': MinioParquetIOManager(s3=minio_resource, bucket_name="oulad-bronze"),
        'silver_io_manager': MinioParquetIOManager(s3=minio_resource, bucket_name="oulad-silver"),
        'gold_io_manager': MinioParquetIOManager(s3=minio_resource, bucket_name="oulad-gold"),
        'dbt': DbtCliResource(project_dir=oulad_dbt_project.project_dir)
    },
)
