WITH weeks_per_course AS (
    SELECT
        dc.course_key,
        dc.module_presentation_length,
        gs.week_number
    FROM {{ ref('dim_course') }} dc
    CROSS JOIN LATERAL generate_series(
        0,
        CEIL(dc.module_presentation_length / 7.0)::int
    ) AS gs(week_number)
)

SELECT
    ROW_NUMBER() OVER (ORDER BY course_key, week_number) AS time_key,
    course_key,
    week_number
FROM weeks_per_course