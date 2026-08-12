import pandas as pd
import io
from dagster_aws.s3 import S3Resource
from dagster import asset, AssetExecutionContext, MetadataValue


# hàm chuẩn hóa
def standard_bronze(df):
    if 'id_student' in df.columns:
        df['id_student'] = df['id_student'].astype('string')

    date_columns = [
        'date',
        'date_submitted',
        'date_registration',
        'date_unregistration'
    ]

    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    return df


# hàm đổi df về parquet vào buffer và đưa lên minio
def write_parquet_to_minio(s3: S3Resource, df: pd.DataFrame, key: str) -> int:
    bucket_name = "oulad-bronze"
    client = s3.get_client()

    # Kiểm tra bucket đã tồn tại chưa, chưa thì tạo
    existing_buckets = client.list_buckets().get("Buckets", [])
    bucket_names = [b["Name"] for b in existing_buckets]
    if bucket_name not in bucket_names:
        client.create_bucket(Bucket=bucket_name)

    # Chuyển df thành parquet trong bộ nhớ
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)

    # Đẩy lên MinIO
    client.put_object(Bucket=bucket_name, Key=key, Body=buffer.getvalue())

    return len(df)


# viết 7 asset bronze
BRONZE_CONFIG = {
    "student_info": {
        "input": "data/raw/studentInfo.csv",
        "output": "student_info/student_info.parquet",
    },
    "student_registration": {
        "input": "data/raw/studentRegistration.csv",
        "output": "student_registration/student_registration.parquet",
    },
    "student_assessment": {
        "input": "data/raw/studentAssessment.csv",
        "output": "student_assessment/student_assessment.parquet",
    },
    "assessments": {
        "input": "data/raw/assessments.csv",
        "output": "assessments/assessments.parquet",
    },
    "courses": {
        "input": "data/raw/courses.csv",
        "output": "courses/courses.parquet",
    },
    "vle": {
        "input": "data/raw/vle.csv",
        "output": "vle/vle.parquet",
    },
    "student_vle": {
        "input": "data/raw/studentVle.csv",
        "output": "student_vle/student_vle.parquet",
    },
}


def process_bronze_asset(
        context: AssetExecutionContext,
        s3: S3Resource,
        config: dict
) -> None:
    df = pd.read_csv(config["input"])

    df = standard_bronze(df)

    rows = write_parquet_to_minio(
        s3,
        df,
        config["output"]
    )

    context.add_output_metadata({
        "rows": MetadataValue.int(rows),
        "source": config["input"],
        "output": config["output"]
    })


# tạo 7 asset bronze
@asset(group_name="bronze")
def bronze_student_info(context: AssetExecutionContext, s3: S3Resource):
    process_bronze_asset(context, s3, BRONZE_CONFIG["student_info"])


@asset(group_name="bronze")
def bronze_student_registration(context: AssetExecutionContext, s3: S3Resource):
    process_bronze_asset(context, s3, BRONZE_CONFIG["student_registration"])


@asset(group_name="bronze")
def bronze_student_assessment(context: AssetExecutionContext, s3: S3Resource):
    process_bronze_asset(context, s3, BRONZE_CONFIG["student_assessment"])


@asset(group_name="bronze")
def bronze_assessments(context: AssetExecutionContext, s3: S3Resource):
    process_bronze_asset(context, s3, BRONZE_CONFIG["assessments"])


@asset(group_name="bronze")
def bronze_courses(context: AssetExecutionContext, s3: S3Resource):
    process_bronze_asset(context, s3, BRONZE_CONFIG["courses"])


@asset(group_name="bronze")
def bronze_vle(context: AssetExecutionContext, s3: S3Resource):
    process_bronze_asset(context, s3, BRONZE_CONFIG["vle"])


@asset(group_name="bronze")
def bronze_student_vle(context: AssetExecutionContext, s3: S3Resource):
    process_bronze_asset(context, s3, BRONZE_CONFIG["student_vle"])





















# # Hiển thị tất cả các cột (không rút gọn thành ...)
# pd.set_option('display.max_columns', None)
#
# # (Tùy chọn) Mở rộng độ rộng dòng để không bị nhảy xuống dòng mới
# pd.set_option('display.width', 1000)
#
# df = pd.read_csv("../../data/raw/courses.csv")
# print(df.head())
#
# df = pd.read_csv("../../data/raw/assessments.csv")
# print("assessments\n", df.head())
# print(df.info())
# df = pd.read_csv("../../data/raw/studentInfo.csv")
# print("studentInfo\n", df.head())
# print(df.info())
# df = pd.read_csv("../../data/raw/studentAssessment.csv")
# print("studentAssessment\n", df.head())
# print(df.info())
# df = pd.read_csv("../../data/raw/studentRegistration.csv")
# print("studentRegistration\n", df.head())
# print(df.info())
# df = pd.read_csv("../../data/raw/studentVle.csv")
# print("studentVle\n", df.head())
# print(df.info())
# df = pd.read_csv("../../data/raw/vle.csv")
# print("vle\n", df.head())
# print(df.info())
