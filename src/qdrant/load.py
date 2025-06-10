import uuid

import pandas as pd
from fastembed import TextEmbedding
from loguru import logger as log
from qdrant_client.models import PointStruct

from src.bigquery.handler import BigQueryHandler
from src.qdrant.base import QdrantBase


class QdrantLoader(QdrantBase):
    def __init__(self, source, collection_name, embedding_column):
        super().__init__(collection_name=collection_name)
        self.source = source
        self.embedding_column = embedding_column
        self.bigquery_client = BigQueryHandler(project_id="tripadvisor-recommendations", credentials_path="./sa.json")
        self.embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def load_data(self):
        log.info(f"Querying source: {self.source}")
        records = self.bigquery_client.fetch_bigquery_as_list(f"SELECT * FROM {self.source}")
        df = pd.DataFrame(records)

        if df.empty:
            log.error("No records found.")
            return

        texts = df[self.embedding_column].dropna().astype(str).tolist()
        log.info(f"Embedding {len(texts)} records...")
        vectors = list(self.embedder.embed(documents=texts, parallel=0))
        log.info("Embedding complete.")

        payloads = df.to_dict(orient="records")
        log.info(f"Upserting {len(payloads)} records to Qdrant...")
        points = [
            PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
            for vector, payload in zip(vectors, payloads)
        ]

        self.client.upsert(collection_name=self.collection_name, points=points)
        log.success(f"Upserted {len(points)} records to Qdrant.")
