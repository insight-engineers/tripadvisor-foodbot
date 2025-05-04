import requests


def encode_url(url_string: str) -> str:
    """
    Encodes a URL string for use in a URL.

    Args:
        url_string (str): The URL string to encode.

    Returns:
        str: The encoded URL string.
    """

    if not url_string:
        return None

    return requests.utils.requote_uri(url_string)
