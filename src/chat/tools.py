import os
from typing import Annotated, Any, Dict, List, Literal

import pandas as pd
import unidecode
from llama_index.core.tools import FunctionTool
from loguru import logger as log
from tabulate import tabulate

from src.chat.client import bigquery_client, core_llm_model
from src.chat.models import RestaurantsFinalized
from src.chat.prompt import ENRICH_PROMPT
from src.helper.utils import encode_url, get_feature_storage_mode
from src.helper.vars import COSINE_THRESHOLD, OPENAI_MODEL, TOP_K
from src.ranker.workflow import build_mcdm_workflow


def candidate_generation_and_ranking(
    english_natural_query: str,
    city_filter: Literal["Ha Noi", "Ho Chi Minh", "Whatever"] = "Whatever",
    kwargs_dict: Dict[str, Any] = None,
) -> pd.DataFrame:
    """
    Retrieves the restaurants that match the user's query.
    The english_natural_query should be a short and concise information containing only the key information (remove speech words or noise, verbs, etc).
    If key information is missing (e.g., city), prompt the user for clarification about location preference, like at Ha Noi or Ho Chi Minh City, etc.
    """
    cosine_threshold: float = COSINE_THRESHOLD
    top_k: int = TOP_K
    decoded_query = unidecode.unidecode(english_natural_query)

    location_top_k = build_mcdm_workflow(top_k, city_filter, cosine_threshold, decoded_query, kwargs_dict)

    if len(location_top_k) < top_k:
        cosine_threshold -= 0.05
        location_top_k = build_mcdm_workflow(top_k, city_filter, cosine_threshold, decoded_query, kwargs_dict)

    return tabulate(location_top_k, headers="keys", tablefmt="github")


def enrich_restaurant_recommendations(
    original_user_message: Annotated[str, "original user message, concat if multiple messages in the conversation"],
    locations: Annotated[List[str], "restaurant ids from the previous function"],
    kwargs_dict: Dict[str, Any] = None,
) -> str:
    """
    Enrich the restaurant recommendations with more information.
    TRUST the previous function, but you can exclude some locations if they are too out of context, acceptable if it's partially relevant.
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
        data_path = os.path.join(project_root, "include", "data", "fs_location.parquet")
        query_result = pd.read_parquet(data_path, engine="pyarrow")
        log.success(f"Loaded {len(query_result)} records from local storage.")
        query_result = query_result[query_result["location_id"].isin([int(loc) for loc in locations])].to_dict("records")
        log.info(f"Filtered records to {len(query_result)} based on provided location IDs.")
    else:
        return "Please contact deve`loper to fix the feature storage mode."

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
            restaurant_parsed = completion.choices[0].message.parsed  # type: RestaurantsFinalized
            restaurant_description = restaurant_parsed.model_dump()
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
                            (f"- [🗺️ Google Maps]({encode_url(loc.get('location_map', ''))})\n" if "location_map" in loc else ""),
                            (f"- [🌐 Restaurant Website]({encode_url(loc.get('location_url', ''))})\n" if "location_url" in loc else ""),
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


candidate_generation_and_ranking_tool = FunctionTool.from_defaults(
    fn=candidate_generation_and_ranking,
    return_direct=False,
)
enrich_restaurant_recommendations_tool = FunctionTool.from_defaults(
    fn=enrich_restaurant_recommendations,
    return_direct=True,
)
