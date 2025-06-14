import base64
import os
import re
import time
from math import atan2, cos, radians, sin, sqrt
from typing import Any, Literal, Tuple

import requests
from dotenv import load_dotenv

from src.helper.vars import CONFIG_FILE, FEATURE_STORAGE_MODE, WELCOME_MESSAGE


def load_dotenv_file(dotenv_path):
    """Load environment variables from a .env file."""
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
    else:
        raise FileNotFoundError(f"{dotenv_path} does not exist.")


def encode_url(url_string: str) -> str:
    """Encodes a URL string for use in a URL."""

    if not url_string:
        return None

    return requests.utils.requote_uri(url_string)


def encode_b64_string(string: str) -> str:
    """Encodes a string to be base64 encoded."""
    if not string:
        return None

    return base64.b64encode(string.encode("utf-8")).decode("utf-8")


def generate_streaming_response(response: Any, delay: float = 0.05):
    """Generator to mimic some LLM response with spaces and newlines preserved"""
    for part in re.findall(r"\S+\s*", str(response)):
        yield part
        time.sleep(delay)


def get_welcome_message(name: str = "food lover") -> str:
    """Returns a welcome message for the chatbot."""
    return WELCOME_MESSAGE.format(name=name)


def get_feature_storage_mode() -> Literal["local", "remote"]:
    """Returns the storage mode for the feature store."""
    return FEATURE_STORAGE_MODE


def get_config_file() -> str:
    """Returns the name of the configuration file."""
    return CONFIG_FILE


def get_admin_account() -> Tuple[str, str, str]:
    """Returns the admin username, password encoded in base64 as tuple and display name."""
    username = os.getenv("CHAINLIT_ADMIN_USERNAME", "admin")
    password = os.getenv("CHAINLIT_ADMIN_PASSWORD", "admin")
    display_name = os.getenv("CHAINLIT_ADMIN_DISPLAY_NAME", "Admin")

    if not password or not username or not display_name:
        raise ValueError("CHAINLIT_ADMIN_USERNAME, CHAINLIT_ADMIN_PASSWORD, or CHAINLIT_ADMIN_DISPLAY_NAME environment variable is not set.")

    return (display_name, encode_b64_string(username), encode_b64_string(password))


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the Haversine distance between two points in kilometers."""
    R = 6371.0  # earth km radius

    # convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance


def normalize_weights(weights_dict):
    """Normalize weights in a dictionary where keys end with '_score' to sum to 1."""
    score_keys = [key for key in weights_dict.keys() if key.endswith("_score")]

    if not score_keys:
        return weights_dict.copy()

    score_values = [weights_dict[key] for key in score_keys]
    total = sum(score_values)
    result = weights_dict.copy()

    if total == 0:
        equal_weight = 1 / len(score_keys)
        for key in score_keys:
            result[key] = equal_weight
    else:
        for key in score_keys:
            result[key] = weights_dict[key] / total

    return result
