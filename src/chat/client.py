import os

from llama_index.llms.openai import OpenAI as AgentOpenAI
from openai import OpenAI as CoreOpenAI

from src.bigquery.handler import BigQueryHandler
from src.qdrant.query import QdrantQuery

agent_llm_model = AgentOpenAI(streaming=False, model="gpt-4o-mini", temperature=0)
core_llm_model = CoreOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


qdrant_client_location = QdrantQuery(
    qdrant_api_url=os.environ.get("QDRANT_API_URL"),
    qdrant_api_key=os.environ.get("QDRANT_API_KEY"),
    collection_name="tripadvisor_locations",
)

qdrant_client_geolocation = QdrantQuery(
    qdrant_api_url=os.environ.get("QDRANT_API_URL"),
    qdrant_api_key=os.environ.get("QDRANT_API_KEY"),
    collection_name="tripadvisor_geolocations",
)

bigquery_client = BigQueryHandler(
    project_id="tripadvisor-recommendations",
    credentials_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "sa.json"),
)
