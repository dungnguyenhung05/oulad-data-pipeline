import io
import pandas as pd
from dagster import ConfigurableIOManager, InputContext, OutputContext
from dagster_aws.s3 import S3Resource


class MinioParquetIOManager(ConfigurableIOManager):
    s3: S3Resource
    bucket_name: str

    def _get_key(self, context) -> str:
        asset_name = context.asset_key.path[-1]
        return f"{asset_name}/{asset_name}.parquet"
    def handle_output(self, context: OutputContext, obj: pd.DataFrame):
        client = self.s3.get_client()
        existing = client.list_buckets().get("Buckets", [])
        if self.bucket_name not in [b["Name"] for b in existing]:
            client.create_bucket(Bucket=self.bucket_name)

        buffer = io.BytesIO()
        obj.to_parquet(buffer, index=False)
        buffer.seek(0)
        client.put_object(Bucket=self.bucket_name, Key=self._get_key(context), Body=buffer.getvalue())
        context.add_output_metadata({"rows": len(obj)})

    def load_input(self, context: InputContext) -> pd.DataFrame:
        client = self.s3.get_client()
        respond = client.get_object(Bucket=self.bucket_name, Key=self._get_key(context))
        return pd.read_parquet(io.BytesIO(respond["Body"].read()))