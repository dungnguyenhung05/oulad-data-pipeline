SELECT
    ROW_NUMBER() OVER (ORDER BY code_module, code_presentation) AS course_key,
    code_module,
    code_presentation,
    module_presentation_length
FROM {{ source('staging_dashboard', 'stg_bronze_courses') }}