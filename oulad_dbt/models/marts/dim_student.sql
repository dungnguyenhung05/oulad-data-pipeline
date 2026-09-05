WITH ranked AS (
    SELECT
        id_student,
        code_module,
        code_presentation,
        gender,
        age_band,
        imd_band,
        highest_education,
        region,
        disability,
        ROW_NUMBER() OVER (
            PARTITION BY id_student, code_presentation
            ORDER BY code_module ASC
        ) AS row_num
    FROM {{ source('staging_dashboard', 'stg_bronze_student_info') }}
),

step_1a AS (
    SELECT
        id_student,
        code_presentation,
        gender,
        age_band,
        imd_band,
        highest_education,
        region,
        disability
    FROM ranked
    WHERE row_num = 1
),

compare_with_previous AS (
    SELECT
        *,
        LAG(age_band) OVER (PARTITION BY id_student ORDER BY code_presentation) AS prev_age_band
    FROM step_1a
),

flagged AS (
    SELECT
        *,
        CASE WHEN age_band IS DISTINCT FROM prev_age_band THEN 1 ELSE 0 END AS is_new_version
    FROM compare_with_previous
),
versioned AS(
    SELECT
        *,
        SUM(is_new_version) OVER (
            PARTITION BY id_student
            ORDER BY code_presentation
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS version_number
    FROM flagged
),
grouped AS (
    SELECT
        id_student,
        version_number,
        MAX(gender) AS gender,
        MAX(age_band) AS age_band,
        MAX(imd_band) AS imd_band,
        MAX(highest_education) AS highest_education,
        MAX(region) AS region,
        MAX(disability) AS disability,
        MIN(code_presentation) AS effective_from_presentation,
        MAX(code_presentation) AS effective_to_presentation
    FROM versioned
    GROUP BY id_student, version_number
),
final AS (
    SELECT
        *,
        effective_to_presentation = MAX(effective_to_presentation) OVER (PARTITION BY id_student) AS is_current,
        ROW_NUMBER() OVER (ORDER BY id_student, effective_from_presentation) AS student_key
    FROM grouped
)

SELECT
    student_key,
    id_student,
    gender,
    age_band,
    imd_band,
    highest_education,
    region,
    disability,
    effective_from_presentation,
    effective_to_presentation,
    is_current
FROM final
ORDER BY id_student, effective_from_presentation

