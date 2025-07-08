import os
from abc import ABC

from loguru import logger as log
from qdrant_client import QdrantClient


class QdrantBase(ABC):
    def __init__(
        self,
        qdrant_api_url: str = None,
        qdrant_api_key: str = None,
        collection_name: str = None,
        vector_size: int = 384,
    ):
        """
        Initialize the QdrantBase class.
        :param qdrant_api_url: URL for the Qdrant API.
        :param qdrant_api_key: API key for Qdrant.
        """

        if not qdrant_api_url or not qdrant_api_key:
            self.qdrant_api_url = os.getenv("QDRANT_API_URL")
            self.qdrant_api_key = os.getenv("QDRANT__SERVICE__API_KEY")
        else:
            self.qdrant_api_url = qdrant_api_url
            self.qdrant_api_key = qdrant_api_key

        self.client = self.__initialize_qdrant()
        self.collection_name = collection_name
        self.vector_size = vector_size

        if self.collection_name:
            self.__create_collection_if_not_exists()

    def __initialize_qdrant(self) -> QdrantClient:
        """
        Initialize the Qdrant client.
        """
        try:
            log.info("Initializing Qdrant client...")
            if not self.qdrant_api_url or not self.qdrant_api_key:
                raise ValueError("Qdrant API URL and API key are required.")
            return QdrantClient(
                url=self.qdrant_api_url,
                api_key=self.qdrant_api_key,
            )
        except Exception as e:
            log.error("Failed to initialize Qdrant client: {}", e)
            raise
        finally:
            log.success("Qdrant client initialized successfully.")

    def __create_collection_if_not_exists(self):
        """
        Create a collection in Qdrant if it does not already exist.
        """
        try:
            self.client.get_collection(self.collection_name)
            log.info(f"Collection '{self.collection_name}' already exists.")
        except Exception:
            log.info(f"Collection '{self.collection_name}' does not exist. Creating it...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={"size": self.vector_size, "distance": "Cosine"},
            )
