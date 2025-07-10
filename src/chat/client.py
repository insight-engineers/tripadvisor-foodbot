import os

from llama_index.llms.openai import OpenAI as AgentOpenAI
from openai import AsyncOpenAI
from openai import OpenAI as CoreOpenAI

from src.bigquery.handler import BigQueryHandler
from src.helper.vars import FEATURE_STORAGE_MODE, OPENAI_CONFIG, OPENAI_MODEL
from src.qdrant.query import QdrantQuery

agent_llm_model = AgentOpenAI(**OPENAI_CONFIG, model=OPENAI_MODEL, streaming=False)
core_llm_model = CoreOpenAI(**OPENAI_CONFIG)
async_core_llm_model = AsyncOpenAI(**OPENAI_CONFIG)

qdrant_client_location = QdrantQuery(
    qdrant_api_url=os.environ.get("QDRANT_API_URL"),
    qdrant_api_key=os.environ.get("QDRANT__SERVICE__API_KEY"),
    collection_name="tripadvisor_locations",
)

qdrant_client_geolocation = QdrantQuery(
    qdrant_api_url=os.environ.get("QDRANT_API_URL"),
    qdrant_api_key=os.environ.get("QDRANT__SERVICE__API_KEY"),
    collection_name="tripadvisor_geolocations",
)

if FEATURE_STORAGE_MODE == "local":
    bigquery_client = None
else:
    bigquery_client = BigQueryHandler(
        project_id="tripadvisor-recommendations",
        credentials_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "sa.json"),
    )
