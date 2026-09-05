-- Dam bao khong co sinh vien co demographic khac nhau trong cung 1 code_presentation (ngoai id_stdent = 685015)

SELECT
    id_student,
    code_presentation,
    COUNT(DISTINCT gender) AS n_gender,
    COUNT(DISTINCT age_band) AS n_age_band,
    COUNT(DISTINCT imd_band) AS n_imd_band,
    COUNT(DISTINCT highest_education) AS n_education,
    COUNT(DISTINCT region) AS n_region,
    COUNT(DISTINCT disability) AS n_disability
FROM {{ source('staging_dashboard', 'stg_bronze_student_info') }}
WHERE id_student != '685015'
GROUP BY id_student, code_presentation
HAVING COUNT(DISTINCT gender) > 1
    OR COUNT(DISTINCT age_band) > 1
    OR COUNT(DISTINCT imd_band) > 1
    OR COUNT(DISTINCT highest_education) > 1
    OR COUNT(DISTINCT region) > 1
    OR COUNT(DISTINCT disability) > 1