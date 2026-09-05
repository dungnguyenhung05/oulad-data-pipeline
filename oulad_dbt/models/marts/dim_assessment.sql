SELECT
    ROW_NUMBER() OVER (ORDER BY id_assessment) AS assessment_key,
    a.id_assessment,
    a.assessment_type,
    a.weight,
    a.date,
    dc.course_key
FROM {{ source('staging_dashboard', 'stg_bronze_assessments') }} a
JOIN {{ref('dim_course')}} dc
    ON a.code_module = dc.code_module
    AND a.code_presentation = dc.code_presentation