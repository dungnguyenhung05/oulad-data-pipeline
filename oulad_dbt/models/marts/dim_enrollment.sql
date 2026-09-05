SELECT
    ROW_NUMBER() OVER (ORDER BY r.id_student, r.code_module, r.code_presentation) AS enrollment_key,
    ds.student_key,
    dc.course_key,
    r.date_registration,
    r.date_unregistration,
    si.num_of_prev_attempts,
    si.studied_credits
FROM {{ source('staging_dashboard', 'stg_bronze_student_registration') }} r
JOIN {{ ref('dim_student') }} ds
    ON r.id_student = ds.id_student
    AND r.code_presentation BETWEEN ds.effective_from_presentation AND ds.effective_to_presentation
JOIN {{ ref('dim_course') }} dc
    ON r.code_module = dc.code_module
    AND r.code_presentation = dc.code_presentation
JOIN {{ source('staging_dashboard', 'stg_bronze_student_info') }} si
    ON r.id_student = si.id_student
    AND r.code_module = si.code_module
    AND r.code_presentation = si.code_presentation