import time

import pandas as pd
from fastembed import TextEmbedding

from src.helper.vars import EMBEDDER_MODEL_NAME
from src.qdrant.base import QdrantBase


class QdrantQuery(QdrantBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedder = TextEmbedding(model_name=EMBEDDER_MODEL_NAME)
        self.selected_columns = [
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

    def search_restaurants(
        self,
        natural_query: str,
        city: str = None,
        limit: int = 10,
        score_threshold: float = 0.5,
    ) -> pd.DataFrame:
        try:
            vector = list(self.embedder.embed([natural_query]))[0]
            filters = []
            if city:
                filters.append({"key": "city", "match": {"value": city}})

            query_filter = {
                "must": filters,
                "must_not": [
                    {"key": "review_count", "match": {"value": 0}},
                    {"key": "review_count", "match": {"value": 1}},
                ],
            }
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=True,
                score_threshold=score_threshold,
            )

            restaurant_list = [
                {
                    **{key: hit.payload.get(key) for key in self.selected_columns},
                    "query_matching_score": hit.score,
                }
                for hit in search_result
            ]

            return pd.DataFrame(restaurant_list)
        except Exception as e:
            raise RuntimeError(f"Error searching restaurants: {e}")
        finally:
            time.sleep(1.5)
