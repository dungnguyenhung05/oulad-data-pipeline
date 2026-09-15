from pathlib import Path

from dagster import AssetExecutionContext, AssetKey
from dagster_dbt import DbtCliResource, DbtProject, dbt_assets, DagsterDbtTranslator


OULAD_DBT_PROJECT_DIR = Path(__file__).joinpath("..", "..", "..", "oulad_dbt").resolve()

oulad_dbt_project = DbtProject(project_dir=OULAD_DBT_PROJECT_DIR)
oulad_dbt_project.prepare_if_dev()


class CustomDagsterDbtTranslator(DagsterDbtTranslator):
    def get_asset_key(self, dbt_resource_props):
        resource_type = dbt_resource_props["resource_type"]

        if resource_type == "source":
            source_name = dbt_resource_props["name"]

            mapping = {
                "bronze_student_info": "stg_bronze_student_info",
                "bronze_courses": "stg_bronze_courses",
                "bronze_assessments": "stg_bronze_assessments",
                "bronze_student_registration": "stg_bronze_student_registration",
                "silver_student_vle_enriched": "stg_silver_student_vle_enriched",
                "silver_student_assessment_enriched": "stg_silver_student_assessment_enriched"
            }

            return AssetKey(mapping.get(source_name, source_name))

        return super().get_asset_key(dbt_resource_props)

    def get_group_name(self, dbt_resource_props: dict) -> str | None:
        return "mart_dashboard"

@dbt_assets(
    manifest=oulad_dbt_project.manifest_path,
    dagster_dbt_translator=CustomDagsterDbtTranslator(),
)
def oulad_dbt_assets(context: AssetExecutionContext, dbt: DbtCliResource):
    yield from dbt.cli(["build"], context=context).stream()