import time
from math import atan2, cos, radians, sin, sqrt

import requests


def encode_url(url_string: str) -> str:
    """Encodes a URL string for use in a URL."""

    if not url_string:
        return None

    return requests.utils.requote_uri(url_string)


def generate_streaming_response(response: str):
    """Generator to mimic some LLM response"""
    for word in response:
        yield word
        time.sleep(0.005)


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
