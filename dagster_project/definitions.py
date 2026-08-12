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

from dagster_project.resources import get_minio_resource

defs = Definitions(
    assets=[
        bronze_student_info,
        bronze_student_registration,
        bronze_student_assessment,
        bronze_assessments,
        bronze_courses,
        bronze_vle,
        bronze_student_vle,
    ],
    resources={
        's3': get_minio_resource()
    },
)