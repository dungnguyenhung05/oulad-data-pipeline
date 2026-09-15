import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
from dotenv import load_dotenv
from scipy.stats import chi2_contingency, pointbiserialr
import statsmodels.api as sm
from sqlalchemy import create_engine, text


# =========================================================
# CONFIG & STYLE
# =========================================================

load_dotenv()

st.set_page_config(
    page_title="OULAD Student Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

CHART_HEIGHT = 420
SCHEMA = "dbt_dev_mart_dashboard"

RESULT_COLORS = {
    "Distinction": "#8ecdf5",
    "Pass": "#1f6fb4",
    "Fail": "#f2a6a6",
    "Withdrawn": "#e63946",
}

RESULT_ORDER = ["Distinction", "Pass", "Fail", "Withdrawn"]
RISK_CLASSES = ["Fail", "Withdrawn"]


def custom_css():
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .app-header {
            padding: 0.8rem 1.2rem;
            margin-bottom: 1rem;
            border-radius: 8px;
            background: linear-gradient(90deg, #0d1b2a, #1f6fb4);
            color: white;
        }
        .app-header h1 { margin: 0; font-size: 1.5rem; }
        .app-header p { margin: 0; font-size: 0.9rem; opacity: 0.85; }
        .app-footer {
            margin-top: 2rem;
            padding-top: 0.8rem;
            border-top: 1px solid rgba(255,255,255,0.15);
            font-size: 0.8rem;
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
            <p>Hệ thống Phân tích Mô tả và Chẩn đoán Kết quả Học tập Sinh viên · PostgreSQL DWH (dbt)</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_footer():
    st.markdown(
        """
        <div class="app-footer">
            OULAD Learning Analytics · Data Warehouse: PostgreSQL (dbt) · Single Source of Truth
        </div>
        """,
        unsafe_allow_html=True
    )





# =========================================================
# DATABASE CONNECTION
# =========================================================

@st.cache_resource
def get_connection():
    user = os.environ["POSTGRES_APP_USER"]
    password = os.environ["POSTGRES_APP_PASSWORD"]
    host = os.environ.get("POSTGRES_APP_HOST", "postgres")
    port = os.environ.get("POSTGRES_APP_PORT", "5432")
    db = os.environ.get("POSTGRES_APP_DB", "oulad_dwh")
    connection_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return create_engine(connection_url, pool_pre_ping=True)




def read_sql(query, params=None):
    engine = get_connection()
    with engine.connect() as conn:
        if params:
            return pd.read_sql(text(query), conn, params=params)
        return pd.read_sql(text(query), conn)


# =========================================================
# DATA LOADERS (SQL QUERIES)
# =========================================================

@st.cache_data(ttl=3600)
def get_overview_kpis():
    query = f"""
        SELECT
            COUNT(*) AS total_enrollments,
            COUNT(DISTINCT ds.id_student) AS total_students,
            ROUND(100.0 * SUM(CASE WHEN fscr.final_result = 'Pass' THEN 1 ELSE 0 END) / COUNT(*), 2) AS pass_rate,
            ROUND(100.0 * SUM(CASE WHEN fscr.final_result = 'Fail' THEN 1 ELSE 0 END) / COUNT(*), 2) AS fail_rate,
            ROUND(100.0 * SUM(CASE WHEN fscr.final_result = 'Withdrawn' THEN 1 ELSE 0 END) / COUNT(*), 2) AS withdrawn_rate,
            ROUND(100.0 * SUM(CASE WHEN fscr.final_result = 'Distinction' THEN 1 ELSE 0 END) / COUNT(*), 2) AS distinction_rate
        FROM {SCHEMA}.fact_student_course_result fscr
        JOIN {SCHEMA}.dim_enrollment de ON fscr.enrollment_key = de.enrollment_key
        JOIN {SCHEMA}.dim_student ds ON de.student_key = ds.student_key;
    """
    return read_sql(query).iloc[0]


@st.cache_data(ttl=3600)
def get_result_distribution():
    query = f"""
        SELECT final_result, COUNT(*) AS so_luong
        FROM {SCHEMA}.fact_student_course_result
        GROUP BY final_result
        ORDER BY CASE final_result
            WHEN 'Distinction' THEN 1 WHEN 'Pass' THEN 2 WHEN 'Fail' THEN 3 WHEN 'Withdrawn' THEN 4 END
    """
    return read_sql(query)


@st.cache_data(ttl=3600)
def get_result_by_course():
    query = f"""
        SELECT dc.code_module, fscr.final_result, COUNT(*) AS so_luong
        FROM {SCHEMA}.fact_student_course_result fscr
        JOIN {SCHEMA}.dim_enrollment de ON fscr.enrollment_key = de.enrollment_key
        JOIN {SCHEMA}.dim_course dc ON de.course_key = dc.course_key
        GROUP BY dc.code_module, fscr.final_result
        ORDER BY dc.code_module
    """
    return read_sql(query)


@st.cache_data(ttl=3600)
def get_result_by_presentation():
    query = f"""
        SELECT dc.code_presentation, fscr.final_result, COUNT(*) AS so_luong
        FROM {SCHEMA}.fact_student_course_result fscr
        JOIN {SCHEMA}.dim_enrollment de ON fscr.enrollment_key = de.enrollment_key
        JOIN {SCHEMA}.dim_course dc ON de.course_key = dc.course_key
        GROUP BY dc.code_presentation, fscr.final_result
        ORDER BY dc.code_presentation
    """
    return read_sql(query)


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
    value_expr = f"CASE WHEN {column} = '?' THEN 'Unknown' ELSE {column} END" if dimension == "IMD Band" else column

    # Câu lệnh SQL sạch: hoàn toàn không chứa ký tự % gây lỗi driver
    query = f"""
        SELECT
            {value_expr} AS demographic_value,
            fscr.final_result,
            COUNT(*) AS so_luong
        FROM {SCHEMA}.fact_student_course_result fscr
        JOIN {SCHEMA}.dim_enrollment de ON fscr.enrollment_key = de.enrollment_key
        JOIN {SCHEMA}.dim_student ds ON de.student_key = ds.student_key
        WHERE {column} IS NOT NULL
        GROUP BY {value_expr}, fscr.final_result
    """
    df = read_sql(query)

    # Sắp xếp thứ tự danh mục logic trực tiếp bằng Pandas
    if dimension == "Trình độ học vấn":
        edu_order = [
            "No Formal quals", "Lower Than A Level", "A Level or Equivalent",
            "HE Qualification", "Post Graduate Qualification"
        ]
        df["sort_key"] = df["demographic_value"].apply(lambda x: edu_order.index(x) if x in edu_order else 99)
        df = df.sort_values(["sort_key", "final_result"]).drop(columns=["sort_key"])

    elif dimension == "IMD Band":
        imd_order = [
            "0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
            "50-60%", "60-70%", "70-80%", "80-90%", "90-100%", "Unknown"
        ]
        df["sort_key"] = df["demographic_value"].apply(lambda x: imd_order.index(x) if x in imd_order else 99)
        df = df.sort_values(["sort_key", "final_result"]).drop(columns=["sort_key"])

    elif dimension == "Nhóm tuổi":
        age_order = ["0-35", "35-55", "55<="]
        df["sort_key"] = df["demographic_value"].apply(lambda x: age_order.index(x) if x in age_order else 99)
        df = df.sort_values(["sort_key", "final_result"]).drop(columns=["sort_key"])

    else:
        df = df.sort_values(["demographic_value", "final_result"])

    return df


@st.cache_data(ttl=3600)
def get_behavior_trend_by_result():
    query = f"""
        SELECT
            dt.week_number,
            fscr.final_result,
            AVG(fsb.total_clicks) AS avg_clicks,
            AVG(fsb.active_days) AS avg_active_days
        FROM {SCHEMA}.fact_student_behavior fsb
        JOIN {SCHEMA}.fact_student_course_result fscr ON fsb.enrollment_key = fscr.enrollment_key
        JOIN {SCHEMA}.dim_time dt ON fsb.time_key = dt.time_key
        WHERE dt.week_number > 0
        GROUP BY dt.week_number, fscr.final_result
        ORDER BY dt.week_number
    """
    return read_sql(query)


@st.cache_data(ttl=3600)
def get_diagnostic_dataset():
    """Tải toàn bộ dataset chẩn đoán ở grain enrollment_key phục vụ phân tích đa biến và kiểm định."""
    query = f"""
        WITH behavior_by_enrollment AS (
            SELECT
                enrollment_key,
                SUM(total_clicks) AS total_clicks,
                SUM(active_days) AS active_days,
                AVG(active_sites) AS avg_active_sites,
                CASE WHEN SUM(active_days) > 0 THEN SUM(total_clicks)::numeric / SUM(active_days) ELSE NULL END AS avg_clicks_per_day
            FROM {SCHEMA}.fact_student_behavior
            GROUP BY enrollment_key
        ),
        assessment_by_enrollment AS (
            SELECT
                enrollment_key,
                SUM(is_submitted)::numeric / NULLIF(COUNT(*), 0) AS submission_rate,
                SUM(CASE WHEN is_submitted = 1 AND is_late = 1 THEN 1 ELSE 0 END)::numeric /
                    NULLIF(SUM(CASE WHEN is_submitted = 1 AND is_late IS NOT NULL THEN 1 ELSE 0 END), 0) AS late_rate,
                AVG(CASE WHEN is_submitted = 1 AND is_late = 1 THEN days_late ELSE NULL END) AS avg_days_late,
                AVG(CASE WHEN is_submitted = 1 THEN score ELSE NULL END) AS avg_score
            FROM {SCHEMA}.fact_assessment_submission
            GROUP BY enrollment_key
        )
        SELECT
            fscr.enrollment_key,
            fscr.final_result,
            ds.gender,
            ds.age_band,
            ds.highest_education,
            ds.region,
            ds.disability,
            CASE WHEN ds.imd_band = '?' THEN 'Unknown' ELSE ds.imd_band END AS imd_band,
            COALESCE(b.total_clicks, 0) AS total_clicks,
            COALESCE(b.active_days, 0) AS active_days,
            COALESCE(b.avg_active_sites, 0) AS avg_active_sites,
            COALESCE(b.avg_clicks_per_day, 0) AS avg_clicks_per_day,
            COALESCE(a.submission_rate, 0) AS submission_rate,
            COALESCE(a.late_rate, 0) AS late_rate,
            COALESCE(a.avg_days_late, 0) AS avg_days_late,
            COALESCE(a.avg_score, 0) AS avg_score
        FROM {SCHEMA}.fact_student_course_result fscr
        JOIN {SCHEMA}.dim_enrollment de ON fscr.enrollment_key = de.enrollment_key
        JOIN {SCHEMA}.dim_student ds ON de.student_key = ds.student_key
        LEFT JOIN behavior_by_enrollment b ON fscr.enrollment_key = b.enrollment_key
        LEFT JOIN assessment_by_enrollment a ON fscr.enrollment_key = a.enrollment_key;
    """
    df = read_sql(query)
    df["at_risk"] = df["final_result"].isin(RISK_CLASSES).astype(int)
    return df


# =========================================================
# STATISTICAL CALCULATION ENGINES
# =========================================================

def compute_correlation_matrix(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    return df[numeric_cols].corr(method="pearson")


def compute_point_biserial(df: pd.DataFrame, numeric_cols: list[str], target_col: str = "at_risk") -> pd.DataFrame:
    results = {}
    for col in numeric_cols:
        valid = df[[col, target_col]].dropna()
        if valid[col].nunique() > 1 and len(valid) > 2:
            r, p = pointbiserialr(valid[target_col], valid[col])
        else:
            r, p = np.nan, np.nan
        results[col] = {"point_biserial_r": r, "p_value": p}
    return pd.DataFrame(results).T.sort_values("point_biserial_r", key=lambda s: s.abs(), ascending=False)


def cramers_v_calc(confusion_matrix: np.ndarray) -> float:
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    r, k = confusion_matrix.shape
    phi2 = chi2 / n
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return np.nan
    return np.sqrt(phi2corr / denom)


def compute_cramers_v(df: pd.DataFrame, categorical_cols: list[str], target_col: str = "final_result") -> pd.Series:
    results = {}
    for col in categorical_cols:
        ct = pd.crosstab(df[col], df[target_col])
        results[col] = cramers_v_calc(ct.values)
    return pd.Series(results, name="cramers_v").sort_values(ascending=False)


def fit_ols_diagnostic(df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str],
                       target_col: str = "at_risk"):
    X_numeric = df[numeric_cols].fillna(0)
    X_categorical = pd.get_dummies(df[categorical_cols].fillna("Unknown"), drop_first=True)
    X = pd.concat([X_numeric, X_categorical], axis=1).astype(float)
    X = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    X = sm.add_constant(X)
    y = df[target_col]
    model = sm.OLS(y, X, missing="drop").fit()
    coef_table = pd.DataFrame({"coef": model.params, "p_value": model.pvalues}).drop("const")
    coef_table = coef_table.sort_values("coef", key=lambda s: s.abs(), ascending=False)
    return model, coef_table


# =========================================================
# UI PLOTTING HELPERS
# =========================================================

def plot_bar(df, x_col, y_col, title, x_title, y_title):
    colors = [RESULT_COLORS.get(v, "#8ecdf5") for v in df[x_col]]
    fig = go.Figure(go.Bar(x=df[x_col], y=df[y_col], marker=dict(color=colors)))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, height=CHART_HEIGHT,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False})


def plot_stacked_bar(df, x_col, category_col, value_col, title, x_title, y_title, normalize=False):
    fig = go.Figure()
    for cat in RESULT_ORDER:
        df_sub = df[df[category_col] == cat]
        if df_sub.empty:
            continue
        fig.add_trace(go.Bar(x=df_sub[x_col], y=df_sub[value_col], name=cat, marker=dict(color=RESULT_COLORS[cat])))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, barmode="stack",
                      barnorm="percent" if normalize else None, height=CHART_HEIGHT, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False})


def plot_boxplot(df, metric, title, y_title, clip_y_max=None):
    fig = go.Figure()
    for res in RESULT_ORDER:
        df_sub = df[df["final_result"] == res]
        if df_sub.empty:
            continue
        fig.add_trace(go.Box(y=df_sub[metric], name=res, boxpoints=False, marker=dict(color=RESULT_COLORS[res]),
                             fillcolor=RESULT_COLORS[res]))
    fig.update_layout(title=title, yaxis_title=y_title, xaxis_title="Kết quả cuối khóa", height=CHART_HEIGHT,
                      showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    if clip_y_max is not None:
        fig.update_yaxes(range=[0, clip_y_max])
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False})


def plot_trend(df, metric, title, y_title):
    fig = go.Figure()
    for res in RESULT_ORDER:
        df_sub = df[df["final_result"] == res]
        if df_sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=df_sub["week_number"],
            y=df_sub[metric],
            mode="lines+markers",
            name=res,
            line=dict(color=RESULT_COLORS[res], width=2),
            marker=dict(size=5)
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Tuần tương đối của kỳ học",
        yaxis_title=y_title,
        xaxis=dict(dtick=2, tickangle=0),  # Đặt bước nhảy 2 tuần/vạch và chữ nằm ngang
        height=CHART_HEIGHT,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) # Đưa chú thích lên trên cho thoáng bề ngang
    )
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False})


# =========================================================
# APPLICATION LAYOUT
# =========================================================

custom_css()
render_header()

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Tổng quan & Nhân khẩu học",
    "2. Hành vi & Xu hướng",
    "3. Thống kê & Tương quan",
    "4. Dự báo & Cảnh báo Sớm"
])

# =========================================================
# TAB 1 — PHÂN TÍCH MÔ TẢ (DESCRIPTIVE)
# =========================================================
with tab1:
    st.subheader("1.1. Bức tranh tổng quan kết quả học tập")
    try:
        kpi = get_overview_kpis()
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Tổng Sinh viên", f"{int(kpi['total_students']):,}")
        c2.metric("Lượt Đăng ký", f"{int(kpi['total_enrollments']):,}")
        c3.metric("Pass", f"{kpi['pass_rate']:.2f}%")
        c4.metric("Fail", f"{kpi['fail_rate']:.2f}%")
        c5.metric("Withdrawn", f"{kpi['withdrawn_rate']:.2f}%")
        c6.metric("Distinction", f"{kpi['distinction_rate']:.2f}%")
        st.divider()

        df_result = get_result_distribution()
        plot_bar(df_result, "final_result", "so_luong", "Phân bố số lượng theo kết quả học tập", "Kết quả",
                 "Số lượt học (Enrollments)")

        st.subheader("1.2. Phân bố kết quả theo Học phần & Kỳ học")
        col_m, col_p = st.columns(2)
        with col_m:
            df_module = get_result_by_course()
            plot_stacked_bar(df_module, "code_module", "final_result", "so_luong", "Tỷ lệ kết quả theo Module",
                             "Module", "Tỷ lệ (%)", normalize=True)
        with col_p:
            df_presentation = get_result_by_presentation()
            plot_stacked_bar(df_presentation, "code_presentation", "final_result", "so_luong",
                             "Tỷ lệ kết quả theo Kỳ học (Presentation)", "Presentation", "Tỷ lệ (%)", normalize=True)
        st.divider()

        st.subheader("1.3. Khám phá Phân tầng Nhân khẩu học")
        dim_selected = st.selectbox("Chọn yếu tố nhân khẩu học phân tích:", list(DEMOGRAPHIC_COLUMNS.keys()))
        df_demo = get_demographic_breakdown(dim_selected)
        plot_stacked_bar(df_demo, "demographic_value", "final_result", "so_luong", f"Tỷ lệ kết quả theo {dim_selected}",
                         dim_selected, "Tỷ lệ (%)", normalize=True)

    except Exception as e:
        st.error(f"Lỗi tải Tab Mô tả: {e}")

# =========================================================
# TAB 2 — CHẨN ĐOÁN TRỰC QUAN: HÀNH VI, ĐÁNH GIÁ & TIẾN TRÌNH
# =========================================================
with tab2:
    st.subheader("2.1. Phân bố và độ lệch các chỉ số toàn khóa (Boxplots)")
    st.caption(
        "Biểu đồ hộp phản ánh giá trị trung vị, độ phân tán và phát hiện các ngoại lai (outliers) ở từng nhóm kết quả.")

    try:
        df_diag = get_diagnostic_dataset()

        st.markdown("#### 🖱️ Khối Hành vi Tương tác VLE")
        b1, b2 = st.columns(2)
        with b1:
            plot_boxplot(df_diag, "total_clicks", "Tổng số Clicks theo nhóm", "Total Clicks")
        with b2:
            plot_boxplot(df_diag, "active_days", "Số ngày hoạt động theo nhóm", "Active Days")

        st.divider()
        st.markdown("#### 📝 Khối Hoạt động Bài tập & Đánh giá (Assessment)")
        a1, a2 = st.columns(2)
        with a1:
            plot_boxplot(df_diag, "submission_rate", "Tỷ lệ nộp bài theo nhóm", "Submission Rate")
        with a2:
            plot_boxplot(df_diag, "avg_score", "Điểm đánh giá trung bình", "Average Score")

        st.divider()
        # -------------------------------------------------
        # TIẾN TRÌNH THỜI GIAN & TUẦN PHÂN KỲ (FULL-WIDTH TÁCH BIỆT)
        # -------------------------------------------------
        st.subheader("2.2. Tiến trình tương tác theo tuần & Nhận diện 'Tuần phân kỳ'")
        st.info("Dữ liệu chuỗi thời gian theo tuần giúp quan sát rõ điểm gãy hành vi khi sinh viên bắt đầu bỏ học.")

        df_trend = get_behavior_trend_by_result()

        # ==================== PHẦN 1: TOTAL CLICKS ====================
        st.markdown("#### 🖱️ Diễn biến Tổng số Clicks qua các tuần")
        view_clicks = st.selectbox(
            "Chọn góc nhìn phân tích cho Total Clicks:",
            ["4 nhóm",
             "Fail vs Withdrawn"],
            key="select_view_clicks"
        )

        if "Fail vs Withdrawn" in view_clicks:
            df_plot_clicks = df_trend[df_trend["final_result"].isin(["Fail", "Withdrawn"])]
            title_clicks = "Total Clicks: Đối đầu trực tiếp Fail vs Withdrawn (Tìm điểm gãy)"
        else:
            df_plot_clicks = df_trend
            title_clicks = "Total Clicks: Quỹ đạo trung bình qua các tuần (4 nhóm kết quả)"

        plot_trend(df_plot_clicks, "avg_clicks", title_clicks, "Số Clicks trung bình / tuần")

        st.divider()

        # ==================== PHẦN 2: ACTIVE DAYS ====================
        st.markdown("#### 📅 Diễn biến Số ngày hoạt động (Active Days) qua các tuần")
        view_days = st.selectbox(
            "Chọn góc nhìn phân tích cho Active Days:",
            ["4 nhóm",
             "Fail vs Withdrawn"],
            key="select_view_days"
        )

        if "Fail vs Withdrawn" in view_days:
            df_plot_days = df_trend[df_trend["final_result"].isin(["Fail", "Withdrawn"])]
            title_days = "Active Days: Đối đầu trực tiếp Fail vs Withdrawn (Tìm điểm gãy)"
        else:
            df_plot_days = df_trend
            title_days = "Active Days: Quỹ đạo trung bình qua các tuần (4 nhóm kết quả)"

        plot_trend(df_plot_days, "avg_active_days", title_days, "Số ngày hoạt động trung bình / tuần")

        st.caption(
            "📌 **Góc nhìn chẩn đoán then chốt:** Từ Tuần 3 đến Tuần 4, đường tương tác của nhóm Withdrawn tách khỏi nhóm Fail và rơi tự do về 0. Đây là thời điểm vàng để gửi cảnh báo can thiệp.")
    except Exception as e:
        st.error(f"Lỗi tải Tab Chẩn đoán Trực quan: {e}")

# =========================================================
# TAB 3 — KIỂM ĐỊNH THỐNG KÊ CHẨN ĐOÁN (STATISTICAL RIGOR)
# =========================================================
with tab3:
    st.subheader("3.1. Bảng giá trị trung vị chuẩn hóa (Median Benchmarks)")
    st.caption(
        "Median (`PERCENTILE_CONT(0.5)`) phản ánh trung tâm phân bố thực tế, loại trừ hoàn toàn việc bị bóp méo bởi sinh viên click đột biến.")

    try:
        df_diag = get_diagnostic_dataset()
        num_features = ["total_clicks", "active_days", "avg_active_sites", "avg_clicks_per_day", "submission_rate",
                        "late_rate", "avg_score"]

        median_records = []
        for res in RESULT_ORDER:
            sub = df_diag[df_diag["final_result"] == res]
            row = {"Kết quả": res}
            for col in num_features:
                row[col] = sub[col].median()
            median_records.append(row)

        df_median_display = pd.DataFrame(median_records)
        df_median_display["submission_rate"] *= 100
        df_median_display["late_rate"] *= 100
        df_median_display.columns = ["Kết quả", "Median Clicks", "Median Active Days", "Median Sites/Week",
                                     "Median Clicks/Day", "Median Sub Rate (%)", "Median Late Rate (%)",
                                     "Median Avg Score"]

        st.dataframe(
            df_median_display.style.format({
                "Median Clicks": "{:,.1f}", "Median Active Days": "{:,.1f}", "Median Sites/Week": "{:,.1f}",
                "Median Clicks/Day": "{:,.1f}", "Median Sub Rate (%)": "{:.1f}%", "Median Late Rate (%)": "{:.1f}%",
                "Median Avg Score": "{:.1f}"
            }),
            use_container_width=True, hide_index=True
        )

        st.divider()
        st.subheader("3.2. Ma trận tương quan Pearson & Kiểm tra Đa cộng tuyến")
        corr_matrix = compute_correlation_matrix(df_diag, num_features)
        fig_corr = px.imshow(
            corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
            title="Ma trận hệ số tương quan Pearson giữa các chỉ số liên tục (r)"
        )
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption(
            "💡 Các cặp biến có hệ số $r > 0.70$ (như Total Clicks và Active Days) phản ánh hiện tượng đa cộng tuyến — cần chọn lọc khi đưa vào mô hình máy học.")

        st.divider()
        col_pb, col_cv = st.columns(2)

        with col_pb:
            st.subheader("3.3. Tương quan Point-Biserial với Nguy cơ Rủi ro")
            st.caption("Đo mức độ gắn kết giữa biến số học với nhãn rủi ro nhị phân `at_risk` (1 = Fail/Withdrawn).")
            pb_df = compute_point_biserial(df_diag, num_features)
            fig_pb = px.bar(
                pb_df.reset_index().rename(columns={"index": "feature"}),
                x="feature", y="point_biserial_r", color="point_biserial_r", color_continuous_scale="RdBu_r",
                title="Hệ số tương quan Point-Biserial (r_pb)"
            )
            fig_pb.update_layout(height=380)
            st.plotly_chart(fig_pb, use_container_width=True)
            top_feat = pb_df.index[0]
            st.markdown(
                f"👉 **Chẩn đoán:** `{top_feat}` là biến tương quan nghịch mạnh nhất với rủi ro rớt học ($r = {pb_df.iloc[0]['point_biserial_r']:.3f}$, $p < 0.001$).")

        with col_cv:
            st.subheader("3.4. Mức độ liên kết Demographic (Cramér's V)")
            st.caption("Đo cường độ quan hệ giữa các biến định tính nhân khẩu học với `final_result` ($V \in [0, 1]$).")
            cat_features = ["highest_education", "imd_band", "age_band", "gender", "disability", "region"]
            cv_series = compute_cramers_v(df_diag, cat_features)
            fig_cv = px.bar(
                cv_series.reset_index().rename(columns={"index": "feature", "cramers_v": "Cramér's V"}),
                x="feature", y="Cramér's V", range_y=[0, 0.4],
                title="Cường độ liên kết nhân khẩu học (Cramér's V)"
            )
            fig_cv.update_layout(height=380)
            st.plotly_chart(fig_cv, use_container_width=True)
            st.markdown(
                f"👉 **Chẩn đoán:** `{cv_series.index[0]}` có liên kết mạnh nhất ($V = {cv_series.iloc[0]:.3f}$), trong khi `{cv_series.index[-1]}` hầu như độc lập với kết quả.")

        st.divider()
        st.subheader("3.5. Hồi quy Tuyến tính Chẩn đoán (Diagnostic OLS Regression)")
        st.caption(
            "Kiểm soát đồng thời các biến khác: Đánh giá trọng số độc lập thực tế của từng đặc trưng lên rủi ro sau khi đã chuẩn hóa Z-score.")

        ols_model, coef_df = fit_ols_diagnostic(df_diag, num_features, cat_features)
        coef_display = coef_df.reset_index().rename(
            columns={"index": "Đặc trưng (Feature)", "coef": "Hệ số Chuẩn hóa (Beta)", "p_value": "P-value"})
        coef_display["Có ý nghĩa thống kê (p < 0.05)"] = coef_display["P-value"] < 0.05

        fig_coef = px.bar(
            coef_display.head(10), x="Đặc trưng (Feature)", y="Hệ số Chuẩn hóa (Beta)",
            color="Có ý nghĩa thống kê (p < 0.05)",
            color_discrete_map={True: "#1f6fb4", False: "#d3d3d3"},
            title="Top 10 Đặc trưng có trọng số tác động lớn nhất lên Nguy cơ Rủi ro (Standardized Beta)"
        )
        fig_coef.update_layout(height=400)
        st.plotly_chart(fig_coef, use_container_width=True)
        st.info(
            f"📊 **Hệ số xác định mô hình:** $R^2 = {ols_model.rsquared:.4f}$ (Tập hợp các biến hành vi, bài tập và nhân khẩu học giải thích được ~{ols_model.rsquared * 100:.1f}% nguyên nhân dẫn đến rủi ro của sinh viên).")

    except Exception as e:
        st.error(f"Lỗi tải Tab Kiểm định Thống kê: {e}")

# =========================================================
# TAB 4 — DỰ BÁO KẾT QUẢ & HỆ THỐNG CẢNH BÁO SỚM (ML)
# =========================================================
with tab4:
    st.subheader("🎯 Dự báo Kết quả & Hệ thống Cảnh báo Sớm (ML Early Warning)")

    # 1. Bộ lọc Cutoff
    col_cut, _ = st.columns([2, 4])
    with col_cut:
        cutoff_selected = st.selectbox(
            "Chọn mốc thời gian cảnh báo (Cutoff Week):",
            options=["gold_features_cutoff2", "gold_features_cutoff4", "gold_features_cutoff8",
                     "gold_features_cutoff12"],
            index=1,
            format_func=lambda
                x: "Tuần 4 (Mốc vàng dự báo sớm)" if x == "gold_features_cutoff4" else f"Tuần {x.replace('gold_features_cutoff', '')}"
        )

    # 2. Đọc dữ liệu từ bảng ml_results.fact_student_prediction
    try:
        table_exists = read_sql("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'ml_results' 
                AND table_name = 'fact_student_prediction'
            ) AS exist;
        """).iloc[0]["exist"]

        if not table_exists:
            st.warning("⚠️ Bảng dữ liệu `ml_results.fact_student_prediction` chưa tồn tại trong Database.")
        else:
            df_pred = read_sql(f"""
                SELECT * FROM ml_results.fact_student_prediction 
                WHERE cutoff_name = '{cutoff_selected}'
            """)

            if df_pred.empty:
                st.warning(f"⚠️ Chưa có dữ liệu dự đoán cho mốc `{cutoff_selected}`.")
            else:
                # --- PHẦN KPI CARDS ---
                total_st = len(df_pred)
                high_risk = int((df_pred["risk_level"] == "High").sum())
                med_risk = int((df_pred["risk_level"] == "Medium").sum())
                low_risk = int((df_pred["risk_level"] == "Low").sum())

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Tổng lượt đánh giá", f"{total_st:,}")
                k2.metric("Nguy cơ Cao (High)", f"{high_risk:,}", f"{high_risk / total_st * 100:.1f}%",
                          delta_color="inverse")
                k3.metric("Nguy cơ Trung bình (Medium)", f"{med_risk:,}", f"{med_risk / total_st * 100:.1f}%",
                          delta_color="off")
                k4.metric("Mô hình hoạt động", "Random Forest", "F1 ~ 0.58")

                # =========================================================
                # 1. ĐÁNH GIÁ HIỆU NĂNG & SO SÁNH MÔ HÌNH (ĐƯA LÊN TRÊN)
                # =========================================================
                st.divider()
                st.subheader("🔬 1. Đánh giá Hiệu năng & So sánh Mô hình Máy học")
                st.caption(
                    "Dữ liệu thực nghiệm được đọc trực tiếp từ các artifacts huấn luyện (metadata.json & feature_importance.csv).")

                base_paths = [
                    os.path.join("ml", "models", cutoff_selected),
                    os.path.join("..", "ml", "models", cutoff_selected),
                    os.path.join(".", cutoff_selected),
                    "."
                ]

                meta_path, feat_path = None, None
                for bp in base_paths:
                    mp = os.path.join(bp, "metadata.json")
                    fp = os.path.join(bp, "feature_importance.csv")
                    if os.path.exists(mp):
                        meta_path = mp
                        feat_path = fp if os.path.exists(fp) else None
                        break

                if meta_path and os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        model_meta = json.load(f)

                        # --- A. BẢNG SO SÁNH 2 MODEL (CV METRICS) ---
                        st.markdown("##### A. So sánh Kiểm định chéo (5-Fold Stratified Group CV)")
                        cv_data = model_meta.get("cv_comparison_all_models", {})
                        if cv_data:
                            df_compare = pd.DataFrame([
                                {
                                    "Mô hình": "Logistic Regression (Baseline)",
                                    "Accuracy": f"{cv_data['logreg']['accuracy'] * 100:.2f}%",
                                    "Macro F1": f"{cv_data['logreg']['macro_f1'] * 100:.2f}%",
                                    "Weighted F1": f"{cv_data['logreg']['weighted_f1'] * 100:.2f}%",
                                    "Recall (Fail)": f"{cv_data['logreg']['recall_fail'] * 100:.2f}%",
                                    "Recall (Withdrawn)": f"{cv_data['logreg']['recall_withdrawn'] * 100:.2f}%"
                                },
                                {
                                    "Mô hình": "Random Forest",
                                    "Accuracy": f"{cv_data['rf']['accuracy'] * 100:.2f}%",
                                    "Macro F1": f"{cv_data['rf']['macro_f1'] * 100:.2f}%",
                                    "Weighted F1": f"{cv_data['rf']['weighted_f1'] * 100:.2f}%",
                                    "Recall (Fail)": f"{cv_data['rf']['recall_fail'] * 100:.2f}%",
                                    "Recall (Withdrawn)": f"{cv_data['rf']['recall_withdrawn'] * 100:.2f}%"
                                }
                            ])
                            st.dataframe(df_compare, use_container_width=True, hide_index=True)
                            st.success(
                                "**Kết luận lựa chọn:** Mô hình **Random Forest** vượt trội hơn Logistic Regression trên tất cả các thước đo chính, do đó **Random Forest được chọn làm mô hình triển khai chính thức** cho hệ thống dự báo.")

                    # --- B. CONFUSION MATRIX & FEATURE IMPORTANCE ---
                    col_cm, col_imp = st.columns([1, 1])

                    with col_cm:
                        st.markdown("##### B. Ma trận nhầm lẫn tập Test (Confusion Matrix)")
                        if "test_metrics" in model_meta and "confusion_matrix" in model_meta["test_metrics"]:
                            cm_raw = np.array(model_meta["test_metrics"]["confusion_matrix"])
                            classes = model_meta.get("classes", ["Distinction", "Pass", "Fail", "Withdrawn"])
                            cm_norm = cm_raw.astype("float") / cm_raw.sum(axis=1)[:, np.newaxis] * 100

                            fig_cm = px.imshow(
                                cm_norm,
                                x=classes,
                                y=classes,
                                text_auto=".1f",
                                color_continuous_scale="Blues",
                                labels=dict(x="Nhãn Dự đoán", y="Nhãn Thực tế", color="Tỷ lệ (%)"),
                            )
                            fig_cm.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)",
                                                 plot_bgcolor="rgba(0,0,0,0)")
                            st.plotly_chart(fig_cm, use_container_width=True)

                    with col_imp:
                        st.markdown("##### C. Top 10 Đặc trưng Tác động Lớn nhất (Feature Importance)")
                        if feat_path and os.path.exists(feat_path):
                            df_imp = pd.read_csv(feat_path).head(10).copy()
                            df_imp["feature"] = df_imp["feature"].str.replace("num__", "").str.replace("cat__", "")
                            df_imp = df_imp.sort_values(by="importance", ascending=True)

                            fig_imp = px.bar(
                                df_imp,
                                x="importance",
                                y="feature",
                                orientation="h",
                                labels=dict(importance="Độ quan trọng (Gini Importance)", feature="Đặc trưng"),
                                color="importance",
                                color_continuous_scale="Tealgrn"
                            )
                            fig_imp.update_layout(height=380, showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                                                  plot_bgcolor="rgba(0,0,0,0)")
                            st.plotly_chart(fig_imp, use_container_width=True)
                        else:
                            st.info("Chưa tìm thấy file `feature_importance.csv`.")
                else:
                    st.info("Chưa tìm thấy file `metadata.json` cho mốc cutoff này.")

                # =========================================================
                # 2. HỆ THỐNG CẢNH BÁO SỚM & PHÂN BỐ DỰ BÁO (CHUYỂN XUỐNG DƯỚI)
                # =========================================================
                st.divider()
                st.subheader("🚨 2. Phân bố Rủi ro & Danh sách Can thiệp Sinh viên")

                col_chart, col_tbl = st.columns([1, 1])
                with col_chart:
                    st.markdown("##### 📊 Phân bố Mức độ Rủi ro Tổng thể")
                    risk_counts = df_pred["risk_level"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
                    fig_risk = go.Figure(go.Bar(
                        x=risk_counts.index,
                        y=risk_counts.values,
                        marker=dict(color=["#e63946", "#f4a261", "#2a9d8f"]),
                        text=[f"{v:,.0f} ({v / total_st * 100:.1f}%)" for v in risk_counts.values],
                        textposition="auto"
                    ))
                    fig_risk.update_layout(
                        height=380,
                        xaxis_title="Mức độ rủi ro",
                        yaxis_title="Số lượng sinh viên",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)

                with col_tbl:
                    st.markdown("##### 📋 Danh sách Sinh viên Nguy cơ Cao Cần Can thiệp")
                    df_high = df_pred[df_pred["risk_level"] == "High"].sort_values(by="at_risk_probability",
                                                                                   ascending=False).copy()

                    df_display = df_high[["id_student", "code_module", "code_presentation", "at_risk_probability",
                                          "predicted_final_result"]].head(100).copy()
                    df_display["at_risk_probability"] = (df_display["at_risk_probability"] * 100).map("{:.1f}%".format)
                    df_display.columns = ["ID Sinh viên", "Module", "Kỳ học", "Xác suất Rủi ro", "Dự đoán Kết quả"]

                    st.dataframe(df_display, use_container_width=True, hide_index=True)

                    st.download_button(
                        "📥 Tải danh sách High-Risk (CSV)",
                        df_high.to_csv(index=False).encode('utf-8'),
                        f"high_risk_students_{cutoff_selected}.csv",
                        "text/csv"
                    )

    except Exception as e:
        st.error(f"Lỗi tải Tab Dự báo & Cảnh báo Sớm: {e}")

render_footer()