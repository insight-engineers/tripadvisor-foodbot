import os
from typing import Dict, List

import boto3
from loguru import logger as log


class S3Client:
    """A client for interacting with an AWS S3 bucket."""

    def __init__(self, bucket_name: str = None):
        """
        Initialize the S3 client with the specified bucket name or from environment variables.
        :param bucket_name: The name of the S3 bucket. If not provided, it will be read from the environment variable AWS_BUCKET_NAME.
        """
        self._required_vars = [
            "AWS_BUCKET_ENDPOINT_URL",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_REGION_NAME",
        ]
        self._bucket_name = bucket_name if bucket_name else os.getenv("AWS_BUCKET_NAME")

        if not self._bucket_name:
            raise ValueError("Bucket name must be provided or set in environment variable AWS_BUCKET_NAME.")

        if not all(os.getenv(var) for var in self._required_vars):
            log.warning("One or more required environment variables are not set: %s", self._required_vars)

        self.s3 = boto3.client(
            service_name="s3",
            endpoint_url=os.getenv("AWS_BUCKET_ENDPOINT_URL"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION_NAME"),
        )
        log.success("S3 client initialized for bucket: {}", self._bucket_name)

    def list_objects(self) -> List[Dict[str, str]]:
        """List all objects in the S3 bucket."""
        response = self.s3.list_objects_v2(Bucket=self._bucket_name)
        objects = response.get("Contents", [])
        return [{"file": obj["Key"], "size": obj["Size"]} for obj in objects]

    def read_object(self, object_key: str) -> str:
        """Read an object from the S3 bucket."""
        response = self.s3.get_object(Bucket=self._bucket_name, Key=object_key)
        return response["Body"].read().decode("utf-8")

    def write_object(self, object_key: str, data: str) -> bool:
        """Write an object to the S3 bucket."""
        response = self.s3.put_object(Bucket=self._bucket_name, Key=object_key, Body=data.encode("utf-8"))
        return response["ResponseMetadata"]["HTTPStatusCode"] == 200

    def upload_file(self, file_path: str) -> bool:
        """Upload a file to the S3 bucket."""
        try:
            object_key = os.path.basename(file_path)
            self.s3.upload_file(file_path, self._bucket_name, object_key)
            return True
        except Exception as e:
            log.error(f"Error uploading file: {e}")
            return False


if __name__ == "__main__":
    bucket_name = os.getenv("AWS_BUCKET_NAME")
    object_key = "secret.yaml"
    s3_client = S3Client(bucket_name)

    # s3_client.upload_file(object_key)
    print(s3_client.list_objects())
    print(s3_client.read_object(object_key))
