import os
import base64

from llama_index.core.tools import FunctionTool

from utils.bigquery import BigQueryHandler
from utils.chat.llm import core_llm_model, llm_settings
from utils.qdrant.query import QdrantQuery

qdrant_client_location = QdrantQuery(
    qdrant_api_url=os.environ.get("QDRANT_API_URL"),
    qdrant_api_key=os.environ.get("QDRANT_API_KEY"),
    collection_name="tripadvisor_locations",
)

bigquery_client = BigQueryHandler(
    project_id="tripadvisor-recommendations",
    credentials_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "sa.json"),
)


import json
import os
from typing import List


def search_and_rank_restaurant(natural_query: str, price_range_filter=None, city_filter=None, top_k=5) -> List[str]:
    """
    This function returns top k restaurants that fit the user's needs. Ask the user if not enough information (like missing price, city,...) given. ASK ONCE BUT INFORMATIVE, GIVE EXAMPLES.

    Params: (if listed specifically, it MUST be one of the options)
        natural_query: Detected query from user, can be enriched or modified for better results. (MUST be in English, translated if needed, correct spelling if needed)
        price_range_filter: [Cheap Eats, Mid-range, Fine Dining]
        city_filter: [Ha Noi, Ho Chi Minh]

    Returns:
        List[str]: List of restaurant ids that match the user's query.

    Rules:
        - Input None to params if no information is given
        - If user mentions number of restaurants, input top_k; otherwise, MUST skip this param
        - MUST input the ids of the restaurants to the next function if this function runs successfully
    """
    locations = qdrant_client_location.search_restaurants(
        natural_query=natural_query,
        top_k=top_k,
        price_range=price_range_filter,
        city=city_filter,
    )

    return [f"{loc['location_id']}" for loc in locations]


def enrich_restaurant_recommendations(original_user_message: str, locations: List[str]) -> str:
    """
    This function finalizes the restaurant recommendations by enriching the data with more information.
    ONLY use this function at the end of the pipeline.

    Params:
        original_user_message: The original user message, concat if multiple messages
        locations: List of restaurant ids that got from the previous function
    """
    query = f"""
    SELECT location_name, address, location_map, location_url, location_image_url, cuisine_list, price_range, location_overall_rate
    FROM `tripadvisor-recommendations.fs_tripadvisor.fs_location`
    WHERE location_id IN ({", ".join(map(str, locations))})
    """

    results = bigquery_client.fetch_bigquery_as_list(query)

    if not results:
        return "No restaurant recommendations found."

    # using llm to make results more informative
    llm_context = [
        {
            "role": "system",
            "content": f"""
                You are a restaurant assistant. You are giving the restaurant recommendations to the user based on the user message and the restaurant data in the JSON format.
                
                Rules that you MUST follow:
                - Make the results more informative and engaging. Answer the user in a friendly and informative manner.
                - Use friendly voice with emojis and fit the user personality.
                - MUST NOT drop any restaurant from the JSON format. Refactor and utilize full information from the JSON format.
                - MUST return the Markdown format. ONLY show image if the image URL is available.
                - MUST answer the user in the language of the user.
                """,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": original_user_message,
                },
                {
                    "type": "input_text",
                    "text": base64.b64encode(json.dumps(results).encode("utf-8")).decode("utf-8"),
                },
            ],
        },
    ]

    llm_response = core_llm_model.responses.create(input=llm_context, **llm_settings)
    return llm_response.output_text


search_and_rank_restaurant_tool = FunctionTool.from_defaults(
    fn=search_and_rank_restaurant,
    return_direct=False,
)
enrich_restaurant_recommendations_tool = FunctionTool.from_defaults(
    fn=enrich_restaurant_recommendations,
    return_direct=True,
)
