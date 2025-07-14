from loguru import logger as log

from src.chat.client import qdrant_client_location
from src.helper.utils import get_central_location_coords, normalize_weights
from src.ranker.electre_iii import build_electre_iii
from src.ranker.scoring import compute_distance_score, compute_normalized_criterion_score


def build_mcdm_workflow(top_k, city_filter, cosine_threshold, query, preferences_dict):
    """
    Builds a multi-criteria decision analysis (MCDA) workflow for ranking restaurants.
    The workflow consists of the following steps:
    1. Search candidate restaurants using vector search
    2. Compute normalized criterion scores for each restaurant
    3. Check user preferences and compute distance score if user has distance preference
    4. Rank restaurants using ELECTRE III algorithm
    """
    search_restaurants_kwargs = {"natural_query": query, "limit": top_k * 100, "score_threshold": cosine_threshold}
    if city_filter in ["Ha Noi", "Ho Chi Minh"]:
        search_restaurants_kwargs["city"] = city_filter
    locations_with_query_matching_score = qdrant_client_location.search_restaurants(**search_restaurants_kwargs)
    log.info("Successfully retrieved candidate restaurants from Qdrant with cosine similarity score")
    if locations_with_query_matching_score.empty:
        raise ValueError(f"No candidate restaurants found for query: {query}. Need to refine query.")
    criteria = ["food", "ambience", "price", "service"]
    locations_with_score = compute_normalized_criterion_score(locations_with_query_matching_score, criteria)
    log.success("Successfully computed normalized criterion scores")
    log.info("Checking user preferences and distance preference")
    user_preferences = preferences_dict.get("user_preferences", {})
    distance_preference = user_preferences.get("distance_preference", False)
    log.info(f"User preferences: {user_preferences}")
    log.info(f"Distance preference: {distance_preference}")
    electre_params = {
        "food_score": {"q": 20, "p": 7.5, "v": 40},
        "ambience_score": {"q": 20, "p": 7.5, "v": 40},
        "price_score": {"q": 20, "p": 7.5, "v": 40},
        "service_score": {"q": 20, "p": 7.5, "v": 40},
    }
    if distance_preference:
        log.info("User has distance preference. Computing distance score")
        central_lat, central_long = get_central_location_coords(city_filter)
        log.info("Computing distance score for restaurants")
        locations_with_score = compute_distance_score(
            locations_with_score,
            user_preferences.get("distance_km"),
            user_lat=central_lat,
            user_long=central_long,
        )
        user_preferences["distance_score"] = 0.1
        electre_params["distance_score"] = {"q": 20, "p": 7.5, "v": 40}
    user_preferences = {key: value for key, value in user_preferences.items() if key in electre_params and isinstance(value, (int, float))}
    user_preferences = normalize_weights(user_preferences)
    log.info(f"Ranking restaurants using ELECTRE III with normalized user preferences: {user_preferences}")
    location_with_electre_rank = build_electre_iii(locations_with_score, user_preferences, electre_params)
    log.success(f"Successfully ranked restaurants using ELECTRE III. Get {top_k} recommendations")
    location_top_k = location_with_electre_rank.head(top_k)
    score_columns = [col for col in location_with_electre_rank.columns if col.endswith("_score")]
    selected_columns = ["location_id", "location_name"] + score_columns
    location_top_k = location_top_k.sort_values(by="electre_rank").reset_index(drop=True)
    location_top_k = location_top_k[selected_columns].drop(columns=["electre_score"], errors="ignore")
    return location_top_k
