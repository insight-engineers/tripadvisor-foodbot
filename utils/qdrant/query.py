from fastembed import TextEmbedding

from utils.qdrant.base import QdrantBase


class QdrantQuery(QdrantBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def search_restaurants(
        self,
        natural_query: str,
        top_k: int = 5,
        price_range: str = None,
        city: str = None,
    ):
        vector = list(self.embedder.embed([natural_query]))[0]

        filters = []
        if price_range:
            filters.append({"key": "price_range", "match": {"value": price_range}})
        if city:
            filters.append({"key": "city", "match": {"value": city}})

        query_filter = {"must": filters} if filters else None

        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=query_filter,
            limit=top_k,
        )

        return [
            {"location_id": hit.payload.get("location_id")} for hit in search_result
        ]

    def search_restaurant_by_review_id(
        self,
        review_id: str,
        top_k: int = 5,
    ):
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=[review_id],
            limit=top_k,
        )

        return [
            {
                "location_name": hit.payload.get("location_name"),
                "address": hit.payload.get("address"),
                "location_map": hit.payload.get("location_map"),
                "location_url": hit.payload.get("location_url"),
                "location_image_url": hit.payload.get("location_image_url"),
                "cuisine_list": hit.payload.get("cuisine_list"),
                "price_range": hit.payload.get("price_range"),
                "location_overall_rate": hit.payload.get("location_overall_rate"),
            }
            for hit in search_result
        ]
