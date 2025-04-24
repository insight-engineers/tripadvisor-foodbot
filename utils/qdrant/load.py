import uuid

from fastembed import TextEmbedding
from loguru import logger as log
from qdrant_client.models import PointStruct

from utils.bigquery import BigQueryHandler
from utils.qdrant.base import QdrantBase


class QdrantLoader(QdrantBase):
    def __init__(self, source, collection_name, embedding_column):
        super().__init__(collection_name=collection_name)
        self.source = source
        self.embedding_column = embedding_column
        self.bigquery_client = BigQueryHandler(
            project_id="tripadvisor-recommendations", credentials_path="./sa.json"
        )
        self.embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def load_data(self):
        log.info(f"Querying source: {self.source}")
        df = self.bigquery_client.fetch_bigquery(f"SELECT * FROM {self.source}")

        if df.empty:
            log.info("No records found.")
            return

        texts = df[self.embedding_column].dropna().astype(str).tolist()
        log.info(f"Embedding {len(texts)} records...")
        vectors = list(self.embedder.embed(documents=texts, parallel=0))
        log.info("Embedding complete.")

        payloads = df.drop(columns=[self.embedding_column]).to_dict(orient="records")
        points = [
            PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
            for vector, payload in zip(vectors, payloads)
        ]

        self.client.upsert(collection_name=self.collection_name, points=points)
        log.success(f"Upserted {len(points)} records to Qdrant.")
