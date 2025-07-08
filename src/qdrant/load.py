import uuid
import warnings

import pandas as pd
from fastembed import TextEmbedding
from loguru import logger as log
from qdrant_client.models import PointStruct
from tqdm.rich import tqdm

from src.bigquery.handler import BigQueryHandler
from src.helper.utils import get_feature_storage_mode
from src.qdrant.base import QdrantBase

warnings.filterwarnings("ignore")


class QdrantLoader(QdrantBase):
    def __init__(self, source, collection_name, embedding_column):
        super().__init__(collection_name=collection_name)
        self.source = source
        self.embedding_column = embedding_column

        if get_feature_storage_mode() != "local":
            self.bigquery_client = BigQueryHandler(project_id="tripadvisor-recommendations", credentials_path="./sa.json")
        else:
            log.warning("Using local storage mode, BigQuery client will not be initialized.")
            self.bigquery_client = None
        self.embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def load_data(self):
        log.info(f"Querying source: {self.source}")

        if self.bigquery_client:
            records = self.bigquery_client.fetch_bigquery_as_list(f"SELECT * FROM {self.source}")
            df = pd.DataFrame(records)
        else:
            df = pd.read_parquet(self.source)  # source must be a local parquet file in local storage mode

        if df.empty:
            log.error("No records found.")
            return

        texts = df[self.embedding_column].dropna().astype(str).tolist()
        log.info(f"Embedding {len(texts)} records...")
        batch_size = 64
        vectors = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i : i + batch_size]
            batch_vectors = list(self.embedder.embed(batch, parallel=0))
            vectors.extend(batch_vectors)
        log.info("Embedding complete.")

        payloads = df.to_dict(orient="records")
        log.info(f"Upserting {len(payloads)} records to Qdrant...")
        points = [PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload) for vector, payload in zip(vectors, payloads)]

        self.client.upsert(collection_name=self.collection_name, points=points)
        log.success(f"Upserted {len(points)} records to Qdrant.")
