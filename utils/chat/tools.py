from typing import Annotated, List, Literal

import unidecode
from llama_index.core.tools import FunctionTool
from loguru import logger as log

from utils.chat.client import bigquery_client, core_llm_model, qdrant_client_location
from utils.chat.prompt import ENRICH_PROMPT
from utils.helper import encode_url


def candidate_generation_and_ranking(
    english_natural_query: Annotated[
        str, "Adjust to ensure grammatical correctness and clarity."
    ],
    price_range_filter: Literal["Cheap Eats", "Mid-range", "Fine Dining"] = None,
    city_filter: Literal["Ha Noi", "Ho Chi Minh"] = None,
) -> List[str]:
    """
    Retrieves the top-K restaurants that match the user's query.
    If key information is missing (e.g., price range, city), prompt the user for clarification.
    Ensure the prompt is clear, concise, and provides examples.
    Do not filter if the user does not provide information.
    """
    log.info("Candidate generation using Qdrant")
    candidate_limit = 10

    locations = qdrant_client_location.search_restaurants(
        natural_query=unidecode.unidecode(english_natural_query),
        candidate_limit=candidate_limit,
        price_range=price_range_filter,
        city=city_filter,
    )

    log.info("Ranking restaurants using model")
    # TODO: Add ranking logic here

    return [f"{loc['location_id']}" for loc in locations]


def enrich_restaurant_recommendations(
    original_user_message: Annotated[
        str, "original user message, concat if multiple messages in the conversation"
    ],
    locations: Annotated[List[str], "restaurant ids from the previous function"],
) -> str:
    """
    Enrich the restaurant recommendations with more information.
    ONLY use this function at the end of the pipeline.
    """
    query = f"""
    SELECT location_name, address, location_map, location_url, image_url, cuisine_list, price_range, location_overall_rate
    FROM `tripadvisor-recommendations.fs_tripadvisor.fs_location`
    WHERE location_id IN ({", ".join(map(str, locations))})
    """

    query_result = bigquery_client.fetch_bigquery_as_list(query)

    if not query_result:
        return "No restaurant recommendations found. Please try again with a different query."
    else:
        check_valid_value = lambda x: (
            True if x is not None and str(x).strip() not in ["", "-1"] else False
        )
        query_result = [
            {k: v for k, v in loc.items() if check_valid_value(v)}
            for loc in query_result
        ]

    restaurant_output = "\n".join(
        [
            f"### **{loc.get('location_name', '')}**\n"
            + "".join(
                [
                    f"- 📍 {loc.get('address', '')}\n" if "address" in loc else "",
                    (
                        f"- ⭐️ {loc.get('location_overall_rate', '')}\n"
                        if "location_overall_rate" in loc
                        and str(loc.get("location_overall_rate")) != "-1"
                        else ""
                    ),
                    (
                        f"- 💸 {loc.get('price_range', '')}\n"
                        if "price_range" in loc
                        and loc.get("price_range") != "Not Defined"
                        else ""
                    ),
                    (
                        f"- [Google Maps]({encode_url(loc.get('location_map', ''))}) | [TripAdvisor]({encode_url(loc.get('location_url', ''))})\n"
                        if "location_map" in loc and "location_url" in loc
                        else ""
                    ),
                    (
                        f"![{loc.get('location_name', '')}]({loc.get('image_url', '')}?w=800&h=500&s=1)\n"
                        if "image_url" in loc
                        else ""
                    ),
                ]
            )
            for loc in query_result
        ]
    )

    llm_context = [
        {
            "role": "developer",
            "content": ENRICH_PROMPT.format(restaurant_data=restaurant_output).strip(),
        },
        {
            "role": "user",
            "content": original_user_message,
        },
    ]

    llm_params = {
        "messages": llm_context,
        "model": "gpt-4o-mini",
        "temperature": 0.15,
        "max_tokens": 4096,
    }

    llm_response = core_llm_model.chat.completions.create(**llm_params)
    llm_output = llm_response.choices[0].message.content
    return llm_output


candidate_generation_and_ranking_tool = FunctionTool.from_defaults(
    fn=candidate_generation_and_ranking,
    return_direct=False,
)
enrich_restaurant_recommendations_tool = FunctionTool.from_defaults(
    fn=enrich_restaurant_recommendations,
    return_direct=True,
)
