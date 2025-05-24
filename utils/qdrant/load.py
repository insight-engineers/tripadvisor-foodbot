import uuid

import pandas as pd
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

        selected_columns = [
            self.embedding_column,
            "location_id",
            "location_name",
            "address",
            "city",
            "location_url",
            "location_map",
            "image_url",
            "image_description",
            "latitude",
            "longitude",
            "location_rank",
            "location_overall_rate",
            "review_count",
            "review_count_scraped",
            "price_range",
            "cuisine_list",
            "food_negative",
            "food_positive",
            "food_neutral",
            "price_negative",
            "price_positive",
            "price_neutral",
            "ambience_negative",
            "ambience_positive",
            "ambience_neutral",
            "service_negative",
            "service_positive",
            "service_neutral",
            "location_negative",
            "location_positive",
            "location_neutral",
            "general_negative",
            "general_positive",
            "general_neutral",
            "friend_type",
            "solo_type",
            "business_type",
            "couple_type",
            "family_type",
        ]
        payloads = df[selected_columns].to_dict(orient="records")
        log.info(f"Upserting {len(payloads)} records to Qdrant...")
        points = [
            PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
            for vector, payload in zip(vectors, payloads)
        ]

        self.client.upsert(collection_name=self.collection_name, points=points)
        log.success(f"Upserted {len(points)} records to Qdrant.")
