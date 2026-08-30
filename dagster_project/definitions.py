from dagster import Definitions

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
    silver_student_registration_clean,
)


from dagster_project.assets.gold import (
    gold_features_cutoff2,
    gold_features_cutoff4,
    gold_features_cutoff8,
    gold_features_cutoff12)


from dagster_project.assets.staging import (
    stg_gold_features_cutoff2,
    stg_gold_features_cutoff4,
    stg_gold_features_cutoff8,
    stg_gold_features_cutoff12
)
from dagster_project.io_manager import MinioParquetIOManager

from dagster_project.resources import get_minio_resource



minio_resource = get_minio_resource()

defs = Definitions(
    assets=[
        bronze_student_info,
        bronze_student_registration,
        bronze_student_assessment,
        bronze_assessments,
        bronze_courses,
        bronze_vle,
        bronze_student_vle,
        silver_student_vle_enriched,
        silver_student_assessment_enriched,
        silver_student_registration_clean,
        gold_features_cutoff2,
        gold_features_cutoff4,
        gold_features_cutoff8,
        gold_features_cutoff12,
        stg_gold_features_cutoff2,
        stg_gold_features_cutoff4,
        stg_gold_features_cutoff8,
        stg_gold_features_cutoff12
    ],
    resources={
        's3': minio_resource,
        'bronze_io_manager': MinioParquetIOManager(s3=minio_resource, bucket_name="oulad-bronze"),
        'silver_io_manager': MinioParquetIOManager(s3=minio_resource, bucket_name="oulad-silver"),
        'gold_io_manager': MinioParquetIOManager(s3=minio_resource, bucket_name="oulad-gold"),
    },
)
