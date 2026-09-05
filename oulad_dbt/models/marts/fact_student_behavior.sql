WITH vle_with_week AS (
    SELECT
        v.id_student,
        v.code_module,
        v.code_presentation,
        CASE WHEN v.date < 0 THEN 0 ELSE FLOOR(v.date / 7) + 1 END AS week_number,
        v.date,
        v.sum_click,
        v.id_site,
        v.activity_type
    FROM {{ source('staging_dashboard', 'stg_silver_student_vle_enriched') }} v
),
engagement AS (
    SELECT
        id_student,
        code_module,
        code_presentation,
        week_number,
        SUM(sum_click) AS total_clicks,
        COUNT(DISTINCT date) AS active_days,
        COUNT(DISTINCT id_site) AS active_sites,
        COUNT(DISTINCT activity_type) AS activity_types
    FROM vle_with_week
    GROUP BY id_student, code_module, code_presentation, week_number
),
engagement_with_keys AS (
    SELECT
        e.enrollment_key,
        dt.time_key,
        eng.total_clicks,
        eng.active_days,
        eng.active_sites,
        eng.activity_types
    FROM engagement eng
    JOIN {{ ref('dim_student') }} ds
        ON eng.id_student = ds.id_student
        AND eng.code_presentation BETWEEN ds.effective_from_presentation AND ds.effective_to_presentation
    JOIN {{ ref('dim_course') }} dc
        ON eng.code_module = dc.code_module
        AND eng.code_presentation = dc.code_presentation
    JOIN {{ ref('dim_enrollment') }} e
        ON e.student_key = ds.student_key
        AND e.course_key = dc.course_key
    JOIN {{ ref('dim_time') }} dt
        ON dt.course_key = dc.course_key
        AND dt.week_number = eng.week_number
),
assessment_agg AS (
    SELECT
        fas.enrollment_key,
        fas.time_key,
        COUNT(*) AS num_assessments_assigned,
        SUM(fas.is_submitted) AS num_assessments_submitted,
        AVG(fas.score) AS avg_assessment_score,
        SUM(CASE WHEN fas.is_late = 1 THEN 1 ELSE 0 END) AS late_submission_count
    FROM {{ ref('fact_assessment_submission') }} fas
    WHERE fas.time_key IS NOT NULL
    GROUP BY fas.enrollment_key, fas.time_key
),
combined AS (
    SELECT
        COALESCE(ew.enrollment_key, aa.enrollment_key) AS enrollment_key,
        COALESCE(ew.time_key, aa.time_key) AS time_key,
        ew.total_clicks,
        ew.active_days,
        ew.active_sites,
        ew.activity_types,
        aa.num_assessments_assigned,
        aa.num_assessments_submitted,
        aa.avg_assessment_score,
        aa.late_submission_count
    FROM engagement_with_keys ew
    FULL OUTER JOIN assessment_agg aa
        ON ew.enrollment_key = aa.enrollment_key
        AND ew.time_key = aa.time_key
)

SELECT
    enrollment_key,
    time_key,
    COALESCE(total_clicks, 0) AS total_clicks,
    COALESCE(active_days, 0) AS active_days,
    COALESCE(active_sites, 0) AS active_sites,
    COALESCE(activity_types, 0) AS activity_types,
    CASE
        WHEN COALESCE(active_days, 0) = 0 THEN NULL
        ELSE COALESCE(total_clicks, 0)::float / active_days
    END AS avg_clicks_per_active_day,
    COALESCE(num_assessments_assigned, 0) AS num_assessments_assigned,
    COALESCE(num_assessments_submitted, 0) AS num_assessments_submitted,
    avg_assessment_score,
    COALESCE(late_submission_count, 0) AS late_submission_count,
    CASE
        WHEN COALESCE(num_assessments_assigned, 0) = 0 THEN NULL
        ELSE COALESCE(num_assessments_submitted, 0)::float / num_assessments_assigned
    END AS submission_rate
FROM combined
