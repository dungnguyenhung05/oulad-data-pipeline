import pandas as pd
from dagster import asset, AssetExecutionContext, MetadataValue



# hàm chuẩn hóa
def standard_bronze(df):
    if 'id_student' in df.columns:
        df['id_student'] = df['id_student'].astype('string')

    if 'score' in df.columns:
        df['score'] = pd.to_numeric(df['score'], errors='coerce')

    if 'week_from' in df.columns:
        df['week_from'] = pd.to_numeric(df['week_from'], errors='coerce')
    if 'week_to' in df.columns:
        df['week_to'] = pd.to_numeric(df['week_to'], errors='coerce')

    if 'imd_band' in df.columns:
        df['imd_band'] = df['imd_band'].replace({"10-20": "10-20%", "?": pd.NA})

    date_columns = [
        'date',
        'date_submitted',
        'date_registration',
        'date_unregistration',
        'week_from',
        'week_to'
    ]

    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    return df

# viết 7 asset bronze
BRONZE_CONFIG = {
    "student_info": {
        "input": "data/raw/studentInfo.csv",
    },
    "student_registration": {
        "input": "data/raw/studentRegistration.csv",
    },
    "student_assessment": {
        "input": "data/raw/studentAssessment.csv",
    },
    "assessments": {
        "input": "data/raw/assessments.csv",
    },
    "courses": {
        "input": "data/raw/courses.csv",
    },
    "vle": {
        "input": "data/raw/vle.csv",
    },
    "student_vle": {
        "input": "data/raw/studentVle.csv",
    },
}


def process_bronze_asset(
        context: AssetExecutionContext,
        config: dict
) -> pd.DataFrame:
    df = pd.read_csv(config["input"])

    df = standard_bronze(df)


    context.add_output_metadata({
        "rows": MetadataValue.int(len(df)),
        "source": config["input"],
    })

    return df


# tạo 7 asset bronze
@asset(group_name="bronze", io_manager_key="bronze_io_manager")
def bronze_student_info(context: AssetExecutionContext) -> pd.DataFrame:
    return process_bronze_asset(context, BRONZE_CONFIG["student_info"])


@asset(group_name="bronze", io_manager_key="bronze_io_manager")
def bronze_student_registration(context: AssetExecutionContext) -> pd.DataFrame:
    return process_bronze_asset(context, BRONZE_CONFIG["student_registration"])


@asset(group_name="bronze", io_manager_key="bronze_io_manager")
def bronze_student_assessment(context: AssetExecutionContext) -> pd.DataFrame:
    return process_bronze_asset(context, BRONZE_CONFIG["student_assessment"])


@asset(group_name="bronze", io_manager_key="bronze_io_manager")
def bronze_assessments(context: AssetExecutionContext) -> pd.DataFrame:
    return process_bronze_asset(context, BRONZE_CONFIG["assessments"])


@asset(group_name="bronze", io_manager_key="bronze_io_manager")
def bronze_courses(context: AssetExecutionContext) -> pd.DataFrame:
    return process_bronze_asset(context, BRONZE_CONFIG["courses"])


@asset(group_name="bronze", io_manager_key="bronze_io_manager")
def bronze_vle(context: AssetExecutionContext) -> pd.DataFrame:
    return process_bronze_asset(context, BRONZE_CONFIG["vle"])


@asset(group_name="bronze", io_manager_key="bronze_io_manager")
def bronze_student_vle(context: AssetExecutionContext) -> pd.DataFrame:
    return process_bronze_asset(context, BRONZE_CONFIG["student_vle"])





















# # Hiển thị tất cả các cột (không rút gọn thành ...)
# pd.set_option('display.max_columns', None)
#
# # (Tùy chọn) Mở rộng độ rộng dòng để không bị nhảy xuống dòng mới
# pd.set_option('display.width', 1000)
#

