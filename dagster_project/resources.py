import os
from dagster_aws.s3 import S3Resource
from sqlalchemy import create_engine

def get_minio_resource() -> S3Resource:
    return S3Resource(
        endpoint_url=os.environ.get('MINIO_ENDPOINT_URL'),
        aws_access_key_id=os.environ['MINIO_ROOT_USER'],
        aws_secret_access_key=os.environ['MINIO_ROOT_PASSWORD'],
    )

def get_postgres_engine():
    host = os.environ["POSTGRES_APP_HOST"]
    port = os.environ["POSTGRES_APP_PORT"]
    db = os.environ["POSTGRES_APP_DB"]
    user = os.environ["POSTGRES_APP_USER"]
    password = os.environ["POSTGRES_APP_PASSWORD"]
    return create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")