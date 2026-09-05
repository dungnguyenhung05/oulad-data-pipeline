WITH base AS (
    SELECT
        e.enrollment_key,
        da.assessment_key,
        da.course_key,
        da.date AS deadline_date
    FROM {{ ref('dim_enrollment') }} e
    JOIN {{ ref('dim_assessment') }} da
        ON e.course_key = da.course_key
),
submissions_with_keys AS (
    SELECT
        ds.student_key,
        da.assessment_key,
        da.course_key,
        s.score,
        s.date_submitted
    FROM {{ source('staging_dashboard', 'stg_silver_student_assessment_enriched') }} s
    JOIN {{ ref('dim_assessment') }} da
        ON s.id_assessment = da.id_assessment
    JOIN {{ ref('dim_student') }} ds
        ON s.id_student = ds.id_student
        AND s.code_presentation BETWEEN ds.effective_from_presentation AND ds.effective_to_presentation
),

submissions_final AS (
    SELECT
        e.enrollment_key,
        sk.assessment_key,
        sk.score,
        sk.date_submitted
    FROM submissions_with_keys sk
    JOIN {{ ref('dim_enrollment') }} e
        ON e.student_key = sk.student_key
        AND e.course_key = sk.course_key
),
final AS (
    SELECT
        b.enrollment_key,
        b.assessment_key,
        CASE WHEN sf.assessment_key IS NOT NULL THEN 1 ELSE 0 END AS is_submitted,
        CASE
            WHEN b.deadline_date IS NULL OR sf.date_submitted IS NULL THEN NULL
            WHEN sf.date_submitted > b.deadline_date THEN 1
            ELSE 0
        END AS is_late,
        sf.score,
        sf.date_submitted AS submission_date,
        CASE
            WHEN b.deadline_date IS NULL OR sf.date_submitted IS NULL THEN NULL
            WHEN sf.date_submitted > b.deadline_date THEN sf.date_submitted - b.deadline_date
            ELSE 0
        END AS days_late,
        dt.time_key
    FROM base b
    LEFT JOIN submissions_final sf
        ON b.enrollment_key = sf.enrollment_key
        AND b.assessment_key = sf.assessment_key
    LEFT JOIN {{ ref('dim_time') }} dt
        ON dt.course_key = b.course_key
        AND dt.week_number = CASE WHEN b.deadline_date < 0 THEN 0 ELSE FLOOR(b.deadline_date / 7) + 1 END
)

SELECT enrollment_key, assessment_key, is_submitted, is_late, score, submission_date, days_late, time_key
FROM final