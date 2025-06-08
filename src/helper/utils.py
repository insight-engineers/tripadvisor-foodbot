import os
import re
import time
from math import atan2, cos, radians, sin, sqrt
from typing import Any

import requests
from dotenv import load_dotenv

from src.chat.prompt import WELCOME_PROMPT


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


def generate_streaming_response(response: Any, delay: float = 0.05):
    """Generator to mimic some LLM response with spaces and newlines preserved"""
    for part in re.findall(r"\S+\s*", str(response)):
        yield part
        time.sleep(delay)


def get_welcome_message(name: str = "food lover") -> str:
    """Returns a welcome message for the chatbot."""
    return WELCOME_PROMPT.format(name=name)


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


def normalize_weights(weights):
    """Normalize a list of weights to sum to 1."""
    total = sum(weights)
    if total == 0:
        return [1 / len(weights)] * len(weights)  # Avoid division by zero
    return [w / total for w in weights] if total > 0 else [0] * len(weights)
