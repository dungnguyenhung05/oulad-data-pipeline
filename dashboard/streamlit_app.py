import os

import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine


# CONFIG


load_dotenv()

st.set_page_config(
    page_title="OULAD Student Analytics Dashboard",
    layout="wide"
)

CHART_HEIGHT = 420

RESULT_COLORS = {
    "Distinction": "#8ecdf5",
    "Pass":        "#1f6fb4",
    "Fail":        "#f2a6a6",
    "Withdrawn":   "#e63946",
}


def custom_css():
    st.markdown(
        """
        <style>
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Header tùy chỉnh */
        .app-header {
            padding: 0.6rem 1rem;
            margin-bottom: 1rem;
            border-radius: 8px;
            background: linear-gradient(90deg, #0d1b2a, #1f6fb4);
            color: white;
        }
        .app-header h1 {
            margin: 0;
            font-size: 1.4rem;
        }
        .app-header p {
            margin: 0;
            font-size: 0.85rem;
            opacity: 0.85;
        }

        /* Footer tùy chỉnh */
        .app-footer {
            margin-top: 2rem;
            padding-top: 0.8rem;
            border-top: 1px solid rgba(255,255,255,0.15);
            font-size: 0.78rem;
            opacity: 0.6;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def render_header():
    st.markdown(
        """
        <div class="app-header">
            <h1>OULAD — Student Analytics Dashboard</h1>
            <p>Phân tích kết quả học tập, hành vi học tập và assessment của sinh viên — nguồn dữ liệu OULAD</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_footer():
    st.markdown(
        """
        <div class="app-footer">
            Module 1 — Student Analytics · Nguồn: mart_dashboard (dbt) · OULAD Data Pipeline & Warehouse
        </div>
        """,
        unsafe_allow_html=True
    )


def render_sidebar():
    with st.sidebar:
        st.header("Thông tin")
        st.caption(
            "Dashboard đọc dữ liệu từ schema "
            f"`{SCHEMA}` (Postgres), xây dựng bằng dbt "
            "theo mô hình Fact Constellation."
        )

        st.divider()

        if st.button("Xóa cache & tải lại dữ liệu"):
            st.cache_data.clear()
            st.rerun()

        st.divider()

        st.caption(
            "Lưu ý: các so sánh trong dashboard chỉ mô tả "
            "sự khác biệt quan sát được giữa các nhóm, "
            "không khẳng định quan hệ nhân quả."
        )

SCHEMA = "dbt_dev_mart_dashboard"


# =========================================================
# DATABASE
# =========================================================

@st.cache_resource
def get_connection():
    user = os.environ["POSTGRES_APP_USER"]
    password = os.environ["POSTGRES_APP_PASSWORD"]
    host = os.environ.get("POSTGRES_APP_HOST", "postgres")
    port = os.environ.get("POSTGRES_APP_PORT", "5432")
    db = os.environ.get("POSTGRES_APP_DB", "oulad_dwh")

    connection_url = (
        f"postgresql+psycopg2://"
        f"{user}:{password}@{host}:{port}/{db}"
    )

    return create_engine(
        connection_url,
        pool_pre_ping=True
    )


# =========================================================
# COMMON
# =========================================================

RESULT_ORDER = [
    "Distinction",
    "Pass",
    "Fail",
    "Withdrawn"
]


def read_sql(query, params=None):
    engine = get_connection()

    with engine.connect() as conn:
        return pd.read_sql(
            query,
            conn,
            params=params
        )


# =========================================================
# TAB 1 — TỔNG QUAN
# =========================================================

@st.cache_data(ttl=3600)
def get_overview_kpis():

    query = f"""
        SELECT
            COUNT(*) AS total_enrollments,

            COUNT(DISTINCT ds.id_student) AS total_students,

            ROUND(
                100.0 * SUM(
                    CASE WHEN fscr.final_result = 'Pass'
                    THEN 1 ELSE 0 END
                ) / COUNT(*),
                2
            ) AS pass_rate,

            ROUND(
                100.0 * SUM(
                    CASE WHEN fscr.final_result = 'Fail'
                    THEN 1 ELSE 0 END
                ) / COUNT(*),
                2
            ) AS fail_rate,

            ROUND(
                100.0 * SUM(
                    CASE WHEN fscr.final_result = 'Withdrawn'
                    THEN 1 ELSE 0 END
                ) / COUNT(*),
                2
            ) AS withdrawn_rate,

            ROUND(
                100.0 * SUM(
                    CASE WHEN fscr.final_result = 'Distinction'
                    THEN 1 ELSE 0 END
                ) / COUNT(*),
                2
            ) AS distinction_rate

        FROM {SCHEMA}.fact_student_course_result fscr
        JOIN {SCHEMA}.dim_enrollment de 
            ON fscr.enrollment_key = de.enrollment_key
        JOIN {SCHEMA}.dim_student ds 
            ON de.student_key = ds.student_key;
    """

    return read_sql(query).iloc[0]


@st.cache_data(ttl=3600)
def get_result_distribution():

    query = f"""
        SELECT
            final_result,
            COUNT(*) AS so_luong
        FROM {SCHEMA}.fact_student_course_result
        GROUP BY final_result
        ORDER BY
            CASE final_result
                WHEN 'Distinction' THEN 1
                WHEN 'Pass' THEN 2
                WHEN 'Fail' THEN 3
                WHEN 'Withdrawn' THEN 4
            END
    """

    return read_sql(query)


@st.cache_data(ttl=3600)
def get_result_by_course():

    query = f"""
        SELECT
            dc.code_module,
            fscr.final_result,
            COUNT(*) AS so_luong
        FROM {SCHEMA}.fact_student_course_result fscr

        JOIN {SCHEMA}.dim_enrollment de
            ON fscr.enrollment_key = de.enrollment_key

        JOIN {SCHEMA}.dim_course dc
            ON de.course_key = dc.course_key

        GROUP BY
            dc.code_module,
            fscr.final_result

        ORDER BY
            dc.code_module
    """

    return read_sql(query)


@st.cache_data(ttl=3600)
def get_result_by_presentation():

    query = f"""
        SELECT
            dc.code_presentation,
            fscr.final_result,
            COUNT(*) AS so_luong
        FROM {SCHEMA}.fact_student_course_result fscr

        JOIN {SCHEMA}.dim_enrollment de
            ON fscr.enrollment_key = de.enrollment_key

        JOIN {SCHEMA}.dim_course dc
            ON de.course_key = dc.course_key

        GROUP BY
            dc.code_presentation,
            fscr.final_result

        ORDER BY
            dc.code_presentation
    """

    return read_sql(query)


# =========================================================
# TAB 2 — NHÂN KHẨU HỌC
# =========================================================

DEMOGRAPHIC_COLUMNS = {
    "Nhóm tuổi": "ds.age_band",
    "Giới tính": "ds.gender",
    "Trình độ học vấn": "ds.highest_education",
    "Khuyết tật": "ds.disability",
    "Khu vực": "ds.region",
    "IMD Band": "ds.imd_band"
}


@st.cache_data(ttl=3600)
def get_demographic_breakdown(dimension):

    column = DEMOGRAPHIC_COLUMNS[dimension]

    value_expr = column
    if dimension == "IMD Band":
        value_expr = f"""
            CASE 
                WHEN {column} = '?' THEN 'Unknown' 
                ELSE {column} 
            END
        """

    order_clause = "demographic_value, fscr.final_result"

    if dimension == "Trình độ học vấn":
        order_clause = f"""
            CASE {column}
                WHEN 'No Formal quals' THEN 1
                WHEN 'Lower Than A Level' THEN 2
                WHEN 'A Level or Equivalent' THEN 3
                WHEN 'HE Qualification' THEN 4
                WHEN 'Post Graduate Qualification' THEN 5
                ELSE 6
            END,
            fscr.final_result
        """
    elif dimension == "IMD Band":
        order_clause = f"""
            CASE {column}
                WHEN '0-10%%' THEN 1
                WHEN '10-20%%' THEN 2
                WHEN '20-30%%' THEN 3
                WHEN '30-40%%' THEN 4
                WHEN '40-50%%' THEN 5
                WHEN '50-60%%' THEN 6
                WHEN '60-70%%' THEN 7
                WHEN '70-80%%' THEN 8
                WHEN '80-90%%' THEN 9
                WHEN '90-100%%' THEN 10
                ELSE 11
            END,
            fscr.final_result
        """
    elif dimension == "Nhóm tuổi":
        order_clause = f"""
            CASE {column}
                WHEN '0-35' THEN 1
                WHEN '35-55' THEN 2
                WHEN '55<=' THEN 3
                ELSE 4
            END,
            fscr.final_result
        """

    query = f"""
        SELECT
            {value_expr} AS demographic_value,
            fscr.final_result,
            COUNT(*) AS so_luong
        FROM {SCHEMA}.fact_student_course_result fscr

        JOIN {SCHEMA}.dim_enrollment de
            ON fscr.enrollment_key = de.enrollment_key

        JOIN {SCHEMA}.dim_student ds
            ON de.student_key = ds.student_key

        WHERE {column} IS NOT NULL

        GROUP BY
            {value_expr},
            {column},
            fscr.final_result

        ORDER BY
            {order_clause}
    """

    return read_sql(query)


# =========================================================
# TAB 3 — HÀNH VI & ASSESSMENT
# =========================================================

@st.cache_data(ttl=3600)
def get_course_behavior_by_result():

    query = f"""
        WITH behavior_by_enrollment AS (

            SELECT
                fsb.enrollment_key,

                SUM(fsb.total_clicks) AS total_clicks,

                SUM(fsb.active_days) AS active_days,

                AVG(fsb.active_sites) AS avg_active_sites,

                CASE
                    WHEN SUM(fsb.active_days) > 0
                    THEN
                        SUM(fsb.total_clicks)::numeric
                        / SUM(fsb.active_days)
                    ELSE NULL
                END AS avg_clicks_per_day

            FROM {SCHEMA}.fact_student_behavior fsb

            GROUP BY
                fsb.enrollment_key
        )

        SELECT
            fscr.final_result,
            b.enrollment_key,
            b.total_clicks,
            b.active_days,
            b.avg_active_sites,
            b.avg_clicks_per_day

        FROM behavior_by_enrollment b

        JOIN {SCHEMA}.fact_student_course_result fscr
            ON b.enrollment_key = fscr.enrollment_key
    """

    return read_sql(query)


@st.cache_data(ttl=3600)
def get_course_assessment_by_result():

    query = f"""
        WITH assessment_by_enrollment AS (

            SELECT
                enrollment_key,

                SUM(is_submitted)::numeric
                / NULLIF(COUNT(*), 0)
                AS submission_rate,

                SUM(
                    CASE
                        WHEN is_submitted = 1
                             AND is_late = 1
                        THEN 1
                        ELSE 0
                    END
                )::numeric
                /
                NULLIF(
                    SUM(
                        CASE
                            WHEN is_submitted = 1
                                 AND is_late IS NOT NULL
                            THEN 1
                            ELSE 0
                        END
                    ),
                    0
                ) AS late_rate,

                AVG(
                    CASE
                        WHEN is_submitted = 1
                             AND is_late = 1
                        THEN days_late
                        ELSE NULL
                    END
                ) AS avg_days_late,

                AVG(
                    CASE
                        WHEN is_submitted = 1
                        THEN score
                        ELSE NULL
                    END
                ) AS avg_score

            FROM {SCHEMA}.fact_assessment_submission

            GROUP BY
                enrollment_key
        )

        SELECT
            fscr.final_result,
            a.enrollment_key,
            a.submission_rate,
            a.late_rate,
            a.avg_days_late,
            a.avg_score

        FROM assessment_by_enrollment a

        JOIN {SCHEMA}.fact_student_course_result fscr
            ON a.enrollment_key = fscr.enrollment_key
    """

    return read_sql(query)


# =========================================================
# TAB 4 — XU HƯỚNG THEO THỜI GIAN
# =========================================================

@st.cache_data(ttl=3600)
def get_behavior_trend_by_result():

    query = f"""
        SELECT
            dt.week_number,
            fscr.final_result,

            AVG(fsb.total_clicks) AS avg_clicks,

            AVG(fsb.active_days) AS avg_active_days,

            AVG(fsb.active_sites) AS avg_active_sites,

            AVG(fsb.submission_rate) AS avg_submission_rate

        FROM {SCHEMA}.fact_student_behavior fsb

        JOIN {SCHEMA}.fact_student_course_result fscr
            ON fsb.enrollment_key = fscr.enrollment_key

        JOIN {SCHEMA}.dim_time dt
            ON fsb.time_key = dt.time_key

        WHERE dt.week_number > 0

        GROUP BY
            dt.week_number,
            fscr.final_result

        ORDER BY
            dt.week_number
    """

    return read_sql(query)


# =========================================================
# TAB 5 — KEY FINDINGS
# =========================================================

@st.cache_data(ttl=3600)
def get_median_findings():

    query = f"""
        WITH behavior_by_enrollment AS (

            SELECT
                enrollment_key,

                SUM(total_clicks) AS total_clicks,

                SUM(active_days) AS active_days,

                AVG(active_sites) AS avg_active_sites,

                CASE
                    WHEN SUM(active_days) > 0
                    THEN
                        SUM(total_clicks)::numeric
                        / SUM(active_days)
                    ELSE NULL
                END AS avg_clicks_per_day

            FROM {SCHEMA}.fact_student_behavior

            GROUP BY enrollment_key
        ),

        assessment_by_enrollment AS (

            SELECT
                enrollment_key,

                SUM(is_submitted)::numeric
                / NULLIF(COUNT(*), 0)
                AS submission_rate,

                SUM(
                    CASE
                        WHEN is_submitted = 1
                             AND is_late = 1
                        THEN 1
                        ELSE 0
                    END
                )::numeric
                /
                NULLIF(
                    SUM(
                        CASE
                            WHEN is_submitted = 1
                                 AND is_late IS NOT NULL
                            THEN 1
                            ELSE 0
                        END
                    ),
                    0
                ) AS late_rate,

                AVG(
                    CASE
                        WHEN is_submitted = 1
                        THEN score
                        ELSE NULL
                    END
                ) AS avg_score

            FROM {SCHEMA}.fact_assessment_submission

            GROUP BY enrollment_key
        ),

        enrollment_features AS (

            SELECT
                fscr.final_result,

                b.total_clicks,
                b.active_days,
                b.avg_active_sites,
                b.avg_clicks_per_day,

                a.submission_rate,
                a.late_rate,
                a.avg_score

            FROM {SCHEMA}.fact_student_course_result fscr

            LEFT JOIN behavior_by_enrollment b
                ON fscr.enrollment_key = b.enrollment_key

            LEFT JOIN assessment_by_enrollment a
                ON fscr.enrollment_key = a.enrollment_key
        )

        SELECT
            final_result,

            PERCENTILE_CONT(0.5)
                WITHIN GROUP (ORDER BY total_clicks)
                AS median_total_clicks,

            PERCENTILE_CONT(0.5)
                WITHIN GROUP (ORDER BY active_days)
                AS median_active_days,

            PERCENTILE_CONT(0.5)
                WITHIN GROUP (ORDER BY avg_active_sites)
                AS median_active_sites,

            PERCENTILE_CONT(0.5)
                WITHIN GROUP (ORDER BY avg_clicks_per_day)
                AS median_avg_clicks_per_day,

            PERCENTILE_CONT(0.5)
                WITHIN GROUP (ORDER BY submission_rate)
                AS median_submission_rate,

            PERCENTILE_CONT(0.5)
                WITHIN GROUP (ORDER BY late_rate)
                AS median_late_rate,

            PERCENTILE_CONT(0.5)
                WITHIN GROUP (ORDER BY avg_score)
                AS median_avg_score

        FROM enrollment_features

        GROUP BY
            final_result

        ORDER BY
            CASE final_result
                WHEN 'Distinction' THEN 1
                WHEN 'Pass' THEN 2
                WHEN 'Fail' THEN 3
                WHEN 'Withdrawn' THEN 4
            END
    """

    return read_sql(query)


# =========================================================
# PLOT FUNCTIONS
# =========================================================

def plot_bar(df, x_col, y_col, title, x_title, y_title):
    colors = [RESULT_COLORS.get(v, "#8ecdf5") for v in df[x_col]]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df[x_col], y=df[y_col], marker=dict(color=colors)))
    fig.update_layout(
        title=title, xaxis_title=x_title, yaxis_title=y_title,
        height=CHART_HEIGHT,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False})


def plot_stacked_bar(df, x_col, category_col, value_col, title, x_title, y_title, normalize=False):
    fig = go.Figure()
    for category in RESULT_ORDER:
        df_sub = df[df[category_col] == category]
        if df_sub.empty:
            continue
        fig.add_trace(go.Bar(
            x=df_sub[x_col], y=df_sub[value_col],
            name=category, marker=dict(color=RESULT_COLORS[category])
        ))
    fig.update_layout(
        title=title, xaxis_title=x_title, yaxis_title=y_title,
        barmode="stack", barnorm="percent" if normalize else None,
        height=CHART_HEIGHT,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False})


def plot_boxplot(df, metric, title, y_title,clip_y_max=None):
    fig = go.Figure()
    for result in RESULT_ORDER:
        df_sub = df[df["final_result"] == result]
        if df_sub.empty:
            continue
        fig.add_trace(go.Box(
            y=df_sub[metric], name=result, boxpoints=False,
            marker=dict(color=RESULT_COLORS[result]),
            fillcolor=RESULT_COLORS[result]
        ))
    fig.update_layout(
        title=title, yaxis_title=y_title, xaxis_title="Kết quả cuối khóa",
        height=CHART_HEIGHT, showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )

    if clip_y_max is not None:
        fig.update_yaxes(range=[0, clip_y_max])
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False})


def plot_trend(df, metric, title, y_title):
    fig = go.Figure()
    for result in RESULT_ORDER:
        df_sub = df[df["final_result"] == result]
        if df_sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=df_sub["week_number"], y=df_sub[metric],
            mode="lines+markers", name=result,
            line=dict(color=RESULT_COLORS[result])
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Tuần tương đối từ khi bắt đầu khóa học",
        yaxis_title=y_title, xaxis=dict(dtick=1),
        height=CHART_HEIGHT,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False})


# =========================================================
# DASHBOARD
# =========================================================

custom_css()
render_sidebar()
render_header()

st.title("OULAD — Student Analytics Dashboard")

st.caption(
    "Phân tích kết quả học tập, hành vi học tập và hoạt động assessment "
    "của sinh viên dựa trên OULAD."
)


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Tổng quan",
        "Nhân khẩu học",
        "Hành vi & Assessment",
        "Xu hướng theo thời gian",
        "Key Findings"
    ]
)


# =========================================================
# TAB 1
# =========================================================

with tab1:

    st.subheader("Tổng quan kết quả học tập")

    try:

        kpi = get_overview_kpis()

        col1, col2, col3, col4, col5, col6 = st.columns(6)

        col1.metric(
            "Sinh viên",
            f"{int(kpi['total_students']):,}"
        )

        col2.metric(
            "Lượt đăng ký",
            f"{int(kpi['total_enrollments']):,}"
        )

        col3.metric(
            "Pass",
            f"{kpi['pass_rate']:.2f}%"
        )

        col4.metric(
            "Fail",
            f"{kpi['fail_rate']:.2f}%"
        )

        col5.metric(
            "Withdrawn",
            f"{kpi['withdrawn_rate']:.2f}%"
        )

        col6.metric(
            "Distinction",
            f"{kpi['distinction_rate']:.2f}%"
        )

        st.divider()

        # -------------------------------------------------
        # Result distribution
        # -------------------------------------------------

        df_result = get_result_distribution()

        plot_bar(
            df_result,
            "final_result",
            "so_luong",
            "Phân bố kết quả học tập",
            "Kết quả",
            "Số lượng enrollment"
        )

        # -------------------------------------------------
        # Result × Module
        # -------------------------------------------------

        st.subheader("Kết quả theo Module")

        df_module = get_result_by_course()

        plot_stacked_bar(
            df_module,
            "code_module",
            "final_result",
            "so_luong",
            "Phân bố kết quả theo Module",
            "Module",
            "Số lượng enrollment",
            normalize=True
        )

        # -------------------------------------------------
        # Result × Presentation
        # -------------------------------------------------

        st.subheader("Kết quả theo Presentation")

        df_presentation = get_result_by_presentation()

        plot_stacked_bar(
            df_presentation,
            "code_presentation",
            "final_result",
            "so_luong",
            "Phân bố kết quả theo Presentation",
            "Presentation",
            "Tỷ lệ (%)",
            normalize=True
        )

    except Exception as e:

        st.error(
            f"Lỗi khi tải dữ liệu Tổng quan: {e}"
        )


# =========================================================
# TAB 2
# =========================================================

with tab2:

    st.subheader("Kết quả học tập theo nhân khẩu học")

    dimension = st.selectbox(
        "Chọn yếu tố nhân khẩu học",
        list(DEMOGRAPHIC_COLUMNS.keys())
    )

    try:

        df_demo = get_demographic_breakdown(
            dimension
        )

        plot_stacked_bar(
            df_demo,
            "demographic_value",
            "final_result",
            "so_luong",
            f"Kết quả theo {dimension}",
            dimension,
            "Tỷ lệ (%)",
            normalize=True
        )

    except Exception as e:

        st.error(
            f"Lỗi khi tải dữ liệu nhân khẩu học: {e}"
        )


# =========================================================
# TAB 3
# =========================================================

with tab3:

    st.subheader(
        "Hành vi học tập và Assessment — toàn khóa"
    )

    # -----------------------------------------------------
    # Behavior
    # -----------------------------------------------------

    st.markdown("### Hành vi học tập")

    try:

        df_behavior = get_course_behavior_by_result()

        col1, col2 = st.columns(2)

        with col1:

            plot_boxplot(
                df_behavior,
                "total_clicks",
                "Tổng số Click theo kết quả",
                "Total Clicks"
            )

        with col2:

            plot_boxplot(
                df_behavior,
                "active_days",
                "Số ngày hoạt động theo kết quả",
                "Active Days"
            )

        col3, col4 = st.columns(2)

        with col3:

            plot_boxplot(
                df_behavior,
                "avg_active_sites",
                "Số Site hoạt động trung bình theo tuần",
                "Average Active Sites / Week"
            )

        with col4:

            plot_boxplot(
                df_behavior,
                "avg_clicks_per_day",
                "Số Click trung bình mỗi ngày hoạt động",
                "Avg Clicks / Active Day"
            )

    except Exception as e:

        st.error(
            f"Lỗi khi tải dữ liệu hành vi: {e}"
        )

    st.divider()

    # -----------------------------------------------------
    # Assessment
    # -----------------------------------------------------

    st.markdown("### Assessment")

    try:

        df_assessment = get_course_assessment_by_result()

        col1, col2 = st.columns(2)

        with col1:

            plot_boxplot(
                df_assessment,
                "submission_rate",
                "Submission Rate theo kết quả",
                "Submission Rate"
            )

        with col2:

            plot_boxplot(
                df_assessment,
                "avg_score",
                "Điểm Assessment trung bình theo kết quả",
                "Average Assessment Score"
            )

        col3, col4 = st.columns(2)

        with col3:

            plot_boxplot(
                df_assessment,
                "late_rate",
                "Late Rate theo kết quả",
                "Late Rate"
            )

        with col4:

            plot_boxplot(
                df_assessment,
                "avg_days_late",
                "Số ngày trễ trung bình theo kết quả (giới hạn hiển thị 30 ngày)",
                "Average Days Late",
                clip_y_max=30
            )

    except Exception as e:

        st.error(
            f"Lỗi khi tải dữ liệu Assessment: {e}"
        )


# =========================================================
# TAB 4
# =========================================================

with tab4:

    st.subheader(
        "Xu hướng hành vi học tập theo thời gian"
    )

    st.info(
        "Các chỉ số trong tab này được tính theo grain "
        "enrollment × tuần từ fact_student_behavior."
    )

    metric_options = {
        "Total Clicks": (
            "avg_clicks",
            "Số Click trung bình / enrollment / tuần"
        ),
        "Active Days": (
            "avg_active_days",
            "Số ngày hoạt động trung bình / tuần"
        ),
        "Active Sites": (
            "avg_active_sites",
            "Số Site hoạt động trung bình / tuần"
        ),
        "Submission Rate": (
            "avg_submission_rate",
            "Submission Rate trung bình / tuần"
        )
    }

    selected_metric = st.selectbox(
        "Chọn chỉ số",
        list(metric_options.keys())
    )

    metric_column, y_title = metric_options[
        selected_metric
    ]

    try:

        df_trend = get_behavior_trend_by_result()

        plot_trend(
            df_trend,
            metric_column,
            f"Xu hướng {selected_metric} theo tuần",
            y_title
        )

        st.divider()

        # -------------------------------------------------
        # Fail vs Withdrawn
        # -------------------------------------------------

        st.subheader(
            "So sánh Fail và Withdrawn"
        )

        df_fw = df_trend[
            df_trend["final_result"].isin(
                ["Fail", "Withdrawn"]
            )
        ]

        plot_trend(
            df_fw,
            metric_column,
            f"{selected_metric}: Fail vs Withdrawn",
            y_title
        )

        st.caption(
            "Mục đích của biểu đồ là quan sát thời điểm "
            "và mức độ khác biệt về hành vi giữa hai nhóm, "
            "không dùng để khẳng định quan hệ nhân quả."
        )

    except Exception as e:

        st.error(
            f"Lỗi khi tải dữ liệu xu hướng: {e}"
        )


# =========================================================
# TAB 5
# =========================================================

with tab5:

    st.subheader(
        "So sánh đặc trưng giữa các nhóm kết quả"
    )

    st.info(
        "Median được sử dụng để mô tả mức trung tâm của "
        "các nhóm và giảm ảnh hưởng của các giá trị ngoại lệ."
    )

    try:

        df_median = get_median_findings()

        display_df = df_median.copy()

        display_df.columns = [
            "Kết quả",
            "Median Total Clicks",
            "Median Active Days",
            "Median Active Sites / Week",
            "Median Avg Clicks / Day",
            "Median Submission Rate",
            "Median Late Rate",
            "Median Avg Score"
        ]

        display_df[
            "Median Submission Rate"
        ] *= 100

        display_df[
            "Median Late Rate"
        ] *= 100

        st.dataframe(
            display_df.style.format({
                "Median Total Clicks": "{:,.2f}",
                "Median Active Days": "{:,.2f}",
                "Median Active Sites / Week": "{:,.2f}",
                "Median Avg Clicks / Day": "{:,.2f}",
                "Median Submission Rate": "{:.2f}%",
                "Median Late Rate": "{:.2f}%",
                "Median Avg Score": "{:.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.divider()

    except Exception as e:

        st.error(
            f"Lỗi khi tải Key Findings: {e}"
        )

render_footer()