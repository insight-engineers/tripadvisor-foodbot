import pandas as pd
import sqlparse
from google.api_core.exceptions import GoogleAPIError
from google.cloud import bigquery
from google.oauth2.service_account import Credentials
from loguru import logger as log


class BigQueryHandler:
    """BigQueryHandler for interacting with BigQuery, including table management, data upload, and fetching queries."""

    def __init__(self, project_id: str, credentials_path: str = None):
        """
        Initialize the BigQueryHandler with a specified project and dataset.

        Args:
            project_id (str): GCP project ID.
            credentials_path (str): Path to the service account JSON file.
        """
        try:
            if not project_id:
                raise ValueError("Project ID is required to initialize BigQueryHandler.")

            if not credentials_path:
                raise ValueError("Please provide a path to the service account JSON file")
            else:
                log.success("Using credentials from: {}", credentials_path)

            self.project_id = project_id
            self.client = bigquery.Client(
                credentials=Credentials.from_service_account_file(credentials_path),
                project=self.project_id,
            )

            log.success("Initialized BigQueryHandler for project: {}", project_id)
        except Exception:
            log.warning("Any query to BigQuery will fail because the BigQueryHandler is not initialized properly.")

    def normalize_query(self, query: str) -> str:
        """
        Normalize a BigQuery query
        E.g: Remove ; at the end or comments

        Args:
            query (str): The query to normalize.
        """
        _query = query.strip().rstrip(";")
        _query = sqlparse.format(
            _query,
            reindent=False,
            strip_comments=True,
            strip_whitespace=True,
            keyword_case="upper",
        )

        log.debug("Normalized query:\n{}".format(_query))
        return _query

    def fetch_bigquery(self, query: str) -> pd.DataFrame:
        """
        Execute a query on a BigQuery table and return the results as a DataFrame.

        Args:
            query (str): The query to execute on the BigQuery table.

        Returns:
            pd.DataFrame: DataFrame containing the query results.
        """
        try:
            _query = self.normalize_query(query)
            dataframe = self.client.query(_query).to_dataframe()

            log.success(
                "Successfully fetched data from src.bigquery. Rows returned: {}",
                len(dataframe),
            )
            return pd.DataFrame(dataframe)

        except GoogleAPIError as api_error:
            log.error("Google API Error during data fetch: {}", api_error)
            raise

        except Exception:
            log.exception("An unexpected error occurred during data fetch.")
            raise

    def fetch_bigquery_as_list(self, query: str) -> list:
        """
        Execute a query on a BigQuery table and return the results as a list of dictionaries.

        Args:
            query (str): The query to execute on the BigQuery table.

        Returns:
            list: List of dictionaries containing the query results.
        """
        try:
            _query = self.normalize_query(query)
            rows = self.client.query(_query).result()
            result = [dict(row.items()) for row in rows]

            log.success(
                "Successfully fetched data from src.bigquery. Rows returned: {}",
                len(result),
            )
            return result

        except GoogleAPIError as api_error:
            log.error("Google API Error during data fetch: {}", api_error)
            raise

        except Exception:
            log.exception("An unexpected error occurred during data fetch.")
            raise

    def upload_parquet_to_bq(self, file_path: str, full_table_id: str, write_disposition="WRITE_TRUNCATE") -> None:
        """
        Upload a Parquet file to a specified BigQuery table.

        Args:
            file_path (str): Path to the Parquet file to upload.
            full_table_id (str): The table name where the data will be uploaded.
            write_disposition (str): Defines the write behavior when data already exists.
                                        Options: 'WRITE_TRUNCATE', 'WRITE_APPEND', 'WRITE_EMPTY'.
                                        Default: 'WRITE_TRUNCATE'.
        """
        try:
            log.info(
                "Starting upload of Parquet file '{}' to BigQuery table '{}'",
                file_path,
                full_table_id,
            )

            job_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.PARQUET,
                write_disposition=write_disposition,
            )

            with open(file_path, "rb") as file:
                load_job = self.client.load_table_from_file(file, full_table_id, job_config=job_config)

            load_job.result()  # Wait for the job to complete.
            log.success(
                "Successfully uploaded file '{}' to table '{}'. Rows loaded: {}",
                file_path,
                full_table_id,
                load_job.output_rows,
            )

        except FileNotFoundError as fnf_error:
            log.error("File not found: {}", fnf_error)
            raise
        except GoogleAPIError as api_error:
            log.error("Google API Error during file upload: {}", api_error)
            raise
        except Exception:
            log.exception("An unexpected error occurred during file upload.")
            raise

    def create_table(self, full_table_id: str, schema: list) -> None:
        """
        Creates a BigQuery table with a specified schema.

        Args:
            full_table_id (str): The table name to create.
            schema (list): List of bigquery.SchemaField objects defining the table schema.
        """
        try:
            log.info(
                "Creating table '{}' with schema: {}",
                full_table_id,
                schema,
            )

            table = bigquery.Table(full_table_id, schema=schema)
            table = self.client.create_table(table)

            log.success("Created table '{}'", full_table_id)

        except GoogleAPIError as api_error:
            log.error("Google API Error during table creation: {}", api_error)
            raise
        except Exception:
            log.exception("An unexpected error occurred during table creation.")
            raise
