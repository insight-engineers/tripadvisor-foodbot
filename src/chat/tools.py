import math
import os
from typing import Annotated, Any, Dict, List, Literal

import pandas as pd
import unidecode
from llama_index.core.tools import FunctionTool
from loguru import logger as log
from tabulate import tabulate

from chat.models import RestaurantsFinalized
from src.chat.client import bigquery_client, core_llm_model, qdrant_client_geolocation, qdrant_client_location
from src.chat.prompt import ENRICH_PROMPT
from src.helper.utils import encode_url, get_candidate_limit, get_feature_storage_mode, normalize_weights
from src.helper.vars import OPENAI_MODEL
from src.ranker.electre_iii import build_electre_iii
from src.ranker.scoring import compute_distance_score, compute_normalized_criterion_score


def scoring_and_ranking(
    english_natural_query: str,
    location_natural_query: str = "",
    city_filter: Literal["Ha Noi", "Ho Chi Minh", "Whatever"] = "Whatever",
    kwargs_dict: Dict[str, Any] = None,
) -> pd.DataFrame:
    """
    Retrieves the restaurants that match the user's query.
    The english_natural_query should be a short and concise information containing only the key information (remove speech words or noise, verbs, etc).
    If key information is missing (e.g., city), prompt the user for clarification about location preference, like at Ha Noi or Ho Chi Minh City, near which ward or district, etc.
    location_natural_query can be skipped input if the user has no preference about the location other than the city.
    """
    log.info("Candidate generation using Qdrant")
    candidate_limit = get_candidate_limit()
    top_k: int = max(1, math.ceil(candidate_limit / 100))
    decoded_query = unidecode.unidecode(english_natural_query)

    search_restaurants_kwargs = {"natural_query": decoded_query}
    search_location_kwargs = {"natural_query": location_natural_query, "limit": 1}

    if city_filter in ["Ha Noi", "Ho Chi Minh"]:
        search_restaurants_kwargs["city"] = city_filter
        search_location_kwargs["city"] = city_filter

    locations_with_query_matching_score = qdrant_client_location.search_restaurants(**search_restaurants_kwargs, limit=candidate_limit)
    log.info("Successfully retrieved candidate restaurants from Qdrant with cosine similarity score")

    criteria = ["food", "ambience", "price", "service"]
    locations_with_score = compute_normalized_criterion_score(locations_with_query_matching_score, criteria)

    log.success("Successfully computed normalized criterion scores")
    log.info("Checking user preferences and distance preference")
    user_preferences = kwargs_dict.get("user_preferences", {})
    distance_preference = user_preferences.get("distance_preference", False)
    log.info(f"User preferences: {user_preferences}")
    log.info(f"Distance preference: {distance_preference}")

    thresholds = {
        "food_score": {"q": 0.05, "p": 0.10, "v": 0.20},
        "ambience_score": {"q": 0.05, "p": 0.10, "v": 0.20},
        "price_score": {"q": 0.05, "p": 0.10, "v": 0.20},
        "service_score": {"q": 0.05, "p": 0.10, "v": 0.20},
        "query_matching_score": {"q": 0.05, "p": 0.10, "v": 0.20},
    }

    if distance_preference:
        log.info("User has distance preference. Get user latitude and longitude")

        query_geolocation = qdrant_client_geolocation.search_lat_long(**search_location_kwargs)
        log.success(f"Successfully retrieved user geolocation from Qdrant: {query_geolocation}")

        query_latitude = query_geolocation[0].get("latitude")
        query_longitude = query_geolocation[0].get("longitude")

        log.info("Computing distance score for restaurants")
        locations_with_score = compute_distance_score(
            locations_with_score,
            user_preferences.get("distance_km"),
            user_lat=query_latitude,
            user_long=query_longitude,
        )
        user_preferences["distance_score"] = 0.25
        thresholds["distance_score"] = {"q": 0.05, "p": 0.10, "v": 0.20}

    user_preferences["query_matching_score"] = 0.5
    user_preferences = {key: value for key, value in user_preferences.items() if key in thresholds and isinstance(value, (int, float))}
    user_preferences = normalize_weights(user_preferences)
    log.info(f"Ranking restaurants using ELECTRE III with normalized user preferences: {user_preferences}")
    location_with_electre_rank = build_electre_iii(locations_with_score, user_preferences, thresholds)

    log.success(f"Successfully ranked restaurants using ELECTRE III. Get {top_k} recommendations")
    location_top_k = location_with_electre_rank.head(top_k)
    score_columns = [col for col in location_with_electre_rank.columns if col.endswith("_score")]
    selected_columns = ["location_id", "location_name"] + score_columns
    location_top_k = location_top_k.sort_values(by="electre_rank").reset_index(drop=True)
    location_top_k = location_top_k[selected_columns].drop(columns=["electre_score"], errors="ignore")

    return tabulate(location_top_k, headers="keys", tablefmt="github")


def enrich_restaurant_recommendations(
    original_user_message: Annotated[str, "original user message, concat if multiple messages in the conversation"],
    locations: Annotated[List[str], "restaurant ids from the previous function"],
    kwargs_dict: Dict[str, Any] = None,
) -> str:
    """
    Enrich the restaurant recommendations with more information.
    TRUST the previous function, MUST get full ids.
    ONLY use this function at the end of the pipeline.
    """
    feature_storage_mode = get_feature_storage_mode()
    if feature_storage_mode == "remote":
        query = f"""
        SELECT location_text_nlp, location_id, location_name, address, location_map, location_url, image_url, cuisine_list, price_range, location_overall_rate
        FROM `tripadvisor-recommendations.fs_tripadvisor.fs_location`
        WHERE location_id IN ({", ".join(map(str, locations))})
        """

        query_result = bigquery_client.fetch_bigquery_as_list(query)
    elif feature_storage_mode == "local":
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        data_path = os.path.join(project_root, "data", "fs_location.parquet")
        query_result = pd.read_parquet(data_path, engine="pyarrow")
        log.success(f"Loaded {len(query_result)} records from local storage.")
        query_result = query_result[query_result["location_id"].isin([int(loc) for loc in locations])].to_dict("records")
        log.info(f"Filtered records to {len(query_result)} based on provided location IDs.")
    else:
        return "Please contact developer to fix the feature storage mode."

    if not query_result:
        return "No restaurant recommendations found. Please try again with a different query."
    else:
        check_valid_value = lambda x: (True if x is not None and str(x).strip() not in ["", "-1"] else False)
        query_result = [{k: v for k, v in loc.items() if check_valid_value(v)} for loc in query_result]

    context_data = [f"user_query: {original_user_message}"]
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

    llm_params = {
        "messages": [
            {
                "role": "developer",
                "content": ENRICH_PROMPT.format(context_data="\n\n".join(context_data)),
            }
        ],
        "model": OPENAI_MODEL,
        "temperature": 0.15,
        "top_p": 0.9,
        "response_format": RestaurantsFinalized,
    }

    try:
        completion = core_llm_model.beta.chat.completions.parse(**llm_params)
        unable_response = RestaurantsFinalized(
            begin_description="",
            restaurants=[],
            end_description_with_follow_up="Unable to generate recommendations at this time.",
        ).model_dump()
        if completion.choices[0].message.parsed:
            restaurant_description = completion.choices[0].message.parsed.model_dump()
        elif completion.choices[0].message.refusal:
            restaurant_description = unable_response
        else:
            restaurant_description = unable_response
    except Exception as e:
        log.error(f"An error occurred: {e}")
        restaurant_description = unable_response

    short_desc_map = {str(item["location_id"]): item["short_description"] for item in restaurant_description.get("restaurants", [])}

    restaurant_output = "\n".join(
        [
            f"{restaurant_description['begin_description']}\n\n---\n",
            "\n".join(
                [
                    f"### **{loc.get('location_name', '')}**\n"
                    + (f"{short_desc_map.get(str(loc.get('location_id')), '')}\n\n" if short_desc_map.get(str(loc.get("location_id"))) else "")
                    + "".join(
                        [
                            (f"- 📍 {loc.get('address', '')}\n" if "address" in loc else ""),
                            (f"- ⭐️ {loc.get('location_overall_rate')}\n" if "location_overall_rate" in loc else ""),
                            (f"- 💰 Price: {loc.get('price_range')}\n" if "price_range" in loc and loc.get("price_range") != "Not Defined" else ""),
                            (f"- [Google Maps]({encode_url(loc.get('location_map', ''))})\n" if "location_map" in loc else ""),
                            (f"- [Restaurant Website]({encode_url(loc.get('location_url', ''))})\n" if "location_url" in loc else ""),
                            (f"![{loc.get('location_name', '')}]({loc.get('image_url', '')}?w=500&h=300&s=1)\n" if "image_url" in loc else ""),
                        ]
                    )
                    for loc in query_result
                ]
            ),
            f"---\n{restaurant_description['end_description_with_follow_up']}\n",
        ]
    )

    return restaurant_output


def candidate_generation(
    kwargs_dict: Dict[str, Any] = None,
) -> int:
    """
    Always use this function ONCE before scoring_and_ranking to get number of candidates.
    """
    log.info("Generating candidate limit for the chatbot")
    candidate_limit = get_candidate_limit()

    if not isinstance(candidate_limit, int) or candidate_limit <= 0:
        log.error(f"Invalid candidate limit: {candidate_limit}. Must be a positive integer.")
        raise ValueError("Candidate limit must be a positive integer.")

    return {"candidate_limit": candidate_limit, "message": "Candidate sets are generated successfully!"}


scoring_and_ranking_tool = FunctionTool.from_defaults(
    fn=scoring_and_ranking,
    return_direct=False,
)
enrich_restaurant_recommendations_tool = FunctionTool.from_defaults(
    fn=enrich_restaurant_recommendations,
    return_direct=True,
)
candidate_generation_tool = FunctionTool.from_defaults(fn=candidate_generation, return_direct=False)
