import os
from dagster_aws.s3 import S3Resource

def get_minio_resource() -> S3Resource:
    return S3Resource(
        endpoint_url=os.environ.get('MINIO_ENDPOINT_URL'),
        aws_access_key_id=os.environ['MINIO_ROOT_USER'],
        aws_secret_access_key=os.environ['MINIO_ROOT_PASSWORD'],
    )