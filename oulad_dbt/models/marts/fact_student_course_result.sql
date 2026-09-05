SELECT
    e.enrollment_key,
    si.final_result,
    CASE WHEN si.final_result = 'Pass' THEN 1 ELSE 0 END AS is_pass,
    CASE WHEN si.final_result = 'Fail' THEN 1 ELSE 0 END AS is_fail,
    CASE WHEN si.final_result = 'Withdrawn' THEN 1 ELSE 0 END AS is_withdrawn,
    CASE WHEN si.final_result = 'Distinction' THEN 1 ELSE 0 END AS is_distinction
FROM {{ source('staging_dashboard', 'stg_bronze_student_info') }} si
JOIN {{ ref('dim_student') }} ds
    ON si.id_student = ds.id_student
    AND si.code_presentation BETWEEN ds.effective_from_presentation AND ds.effective_to_presentation
JOIN {{ ref('dim_course') }} dc
    ON si.code_module = dc.code_module
    AND si.code_presentation = dc.code_presentation
JOIN {{ ref('dim_enrollment') }} e
    ON e.student_key = ds.student_key
    AND e.course_key = dc.course_key