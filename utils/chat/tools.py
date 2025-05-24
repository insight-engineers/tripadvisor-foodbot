import re
from typing import Annotated, Any, Dict, List, Literal

import unidecode
from llama_index.core.tools import FunctionTool
from loguru import logger as log
from pydantic import BaseModel

from ranker.electre_iii import build_electre_iii
from ranker.scoring import compute_distance_score, compute_normalized_criterion_score
from utils.chat.client import bigquery_client, core_llm_model, qdrant_client_location
from utils.chat.prompt import ENRICH_PROMPT
from utils.helper import encode_url


class RestaurantDescription(BaseModel):
    location_id: str
    short_description: str


class RestaurantsFinalized(BaseModel):
    begin_description: str
    restaurants: List[RestaurantDescription]
    end_description: str


def candidate_generation_and_ranking(
    english_natural_query: str,
    city_filter: Literal["Ha Noi", "Ho Chi Minh"] = None,
    kwargs_dict: Dict[str, Any] = None,
) -> List[str]:
    """
    Retrieves the top-K restaurants that match the user's query.
    The english_natural_query should be a short and concise information containing only the key information (remove speech words or noise, verbs, etc).
    If key information is missing (e.g., city), prompt the user for clarification.
    """
    log.info("Candidate generation using Qdrant")
    top_k = 5
    decoded_query = unidecode.unidecode(english_natural_query)
    locations_with_query_matching_score = qdrant_client_location.search_restaurants(
        natural_query=decoded_query, city=city_filter, limit=500
    )

    criteria = ["food", "ambience", "price", "service"]
    locations_with_score = compute_normalized_criterion_score(locations_with_query_matching_score, criteria)

    log.success("Successfully computed normalized criterion scores")
    log.info("Ranking restaurants using ELECTRE III")

    user_preferences = kwargs_dict.get("user_preferences", {})
    distance_preference = kwargs_dict.get("distance_preference", {})
    log.info(f"User preferences: {user_preferences}")
    log.info(f"Distance preference: {distance_preference}")

    if distance_preference:
        log.info("User has distance preference. Computing distance score...")
        locations_with_score = compute_distance_score(
            locations_with_score,
            distance_preference["max_distance"],
            distance_preference["user_lat"],
            distance_preference["user_long"],
        )

    thresholds = {
        "food_score": {"q": 0.05, "p": 0.10, "v": 0.20},
        "ambience_score": {"q": 0.05, "p": 0.10, "v": 0.20},
        "price_score": {"q": 0.05, "p": 0.10, "v": 0.20},
        "service_score": {"q": 0.05, "p": 0.10, "v": 0.20},
        "query_matching_score": {"q": 0.05, "p": 0.10, "v": 0.20},
    }

    if distance_preference:
        thresholds["distance_score"] = {"q": 0.05, "p": 0.10, "v": 0.20}

    user_preferences["query_matching_score"] = 0.5
    location_with_electre_rank = build_electre_iii(locations_with_score, user_preferences, thresholds)

    log.success(
        "Successfully ranked restaurants using ELECTRE III. Found {} restaurants, get {} recommendations",
        len(location_with_electre_rank),
        top_k,
    )

    # convert to list of dict and then get location_id from that
    location_top_k = location_with_electre_rank[:top_k]
    print(location_top_k)
    location_with_electre_rank = location_top_k.to_dict("records")
    return [loc["location_id"] for loc in location_with_electre_rank]


def enrich_restaurant_recommendations(
    original_user_message: Annotated[str, "original user message, concat if multiple messages in the conversation"],
    locations: Annotated[List[str], "restaurant ids from the previous function"],
    kwargs_dict: Dict[str, Any] = None,
) -> str:
    """
    Enrich the restaurant recommendations with more information.
    ONLY use this function at the end of the pipeline.
    """
    query = f"""
    SELECT location_text_nlp, location_id, location_name, address, location_map, location_url, image_url, cuisine_list, price_range, location_overall_rate
    FROM `tripadvisor-recommendations.fs_tripadvisor.fs_location`
    WHERE location_id IN ({", ".join(map(str, locations))})
    """

    query_result = bigquery_client.fetch_bigquery_as_list(query)

    if not query_result:
        return "No restaurant recommendations found. Please try again with a different query."
    else:
        check_valid_value = lambda x: (True if x is not None and str(x).strip() not in ["", "-1"] else False)
        query_result = [{k: v for k, v in loc.items() if check_valid_value(v)} for loc in query_result]

    context_data = [f"User Query: {original_user_message}"]
    context_data.extend(
        [
            str(
                {
                    "location_id": loc.get("location_id", ""),
                    "description": loc.get("short_description", ""),
                }
            )
            for loc in query_result
            if not str(loc.get("short_description")).strip().isspace()
        ]
    )

    llm_context = [
        {
            "role": "developer",
            "content": ENRICH_PROMPT.format(context_data="\n\n".join(context_data)),
        }
    ]

    llm_params = {
        "messages": llm_context,
        "model": "gpt-4.1-nano",
        "temperature": 0.15,
        "top_p": 0.9,
        "response_format": RestaurantsFinalized,
    }

    try:
        completion = core_llm_model.beta.chat.completions.parse(**llm_params)
        unable_response = RestaurantsFinalized(
            begin_description="",
            restaurants=[],
            end_description="Unable to generate recommendations at this time.",
        ).model_dump()
        if completion.choices[0].message.parsed:
            restaurant_description = completion.choices[0].message.parsed.model_dump()
            print("Restaurant description: ", restaurant_description)
        elif completion.choices[0].message.refusal:
            restaurant_description = unable_response
        else:
            restaurant_description = unable_response
    except Exception as e:
        log.error(f"An error occurred: {e}")
        restaurant_description = unable_response

    short_desc_map = {
        str(item["location_id"]): item["short_description"] for item in restaurant_description.get("restaurants", [])
    }

    restaurant_output = "\n".join(
        [
            f"{restaurant_description['begin_description']}\n\n---\n",
            "\n".join(
                [
                    f"### **{loc.get('location_name', '')}**\n"
                    + (
                        f"{short_desc_map.get(str(loc.get('location_id')), '')}\n\n"
                        if short_desc_map.get(str(loc.get("location_id")))
                        else ""
                    )
                    + "".join(
                        [
                            (f"- 📍 {loc.get('address', '')}\n" if "address" in loc else ""),
                            (
                                f"- ⭐️ {loc.get('location_overall_rate', '')}\n"
                                if "location_overall_rate" in loc and str(loc.get("location_overall_rate")) != "-1.0"
                                else ""
                            ),
                            (
                                f"- 💸 {loc.get('price_range', '')}\n"
                                if "price_range" in loc and loc.get("price_range") != "Not Defined"
                                else ""
                            ),
                            (
                                f"- Google Maps: [Google Maps]({encode_url(loc.get('location_map', ''))})\n"
                                if "location_map" in loc
                                else ""
                            ),
                            (
                                f"- Restaurant Website: [Restaurant Website]({encode_url(loc.get('location_url', ''))})\n"
                                if "location_url" in loc
                                else ""
                            ),
                            (
                                f"![{loc.get('location_name', '')}]({loc.get('image_url', '')}?w=500&h=300&s=1)\n"
                                if "image_url" in loc
                                else ""
                            ),
                        ]
                    )
                    for loc in query_result
                ]
            ),
            f"---\n{restaurant_description['end_description']}\n",
        ]
    )

    return restaurant_output


candidate_generation_and_ranking_tool = FunctionTool.from_defaults(
    fn=candidate_generation_and_ranking,
    return_direct=False,
)
enrich_restaurant_recommendations_tool = FunctionTool.from_defaults(
    fn=enrich_restaurant_recommendations,
    return_direct=True,
)
