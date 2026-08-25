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
from dagster_project.io_manager import MinioParquetIOManager

from dagster_project.resources import get_minio_resource

from dagster_project.assets.silver import (
    silver_student_vle_enriched,
    silver_student_assessment_enriched,
    silver_student_registration_clean,
)

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
    ],
    resources={
        's3': get_minio_resource(),
        'bronze_io_manager': MinioParquetIOManager(s3=get_minio_resource(), bucket_name="oulad-bronze"),
        'silver_io_manager': MinioParquetIOManager(s3=get_minio_resource(), bucket_name="oulad-silver")
    },
)
