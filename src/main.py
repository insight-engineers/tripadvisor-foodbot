import os
from typing import Literal, Optional

import chainlit as cl
import chainlit.input_widget as cliw
import yaml
from llama_index.core.callbacks import CallbackManager
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore

from src.chat.agent import ParseParamsAgent
from src.chat.callback import ChainlitStatusCallback
from src.chat.client import agent_llm_model
from src.chat.tools import candidate_generation_and_ranking_tool, enrich_restaurant_recommendations_tool
from src.helper.utils import get_config_file, get_welcome_message
from src.s3.client import S3Client

s3_client = S3Client(bucket_name=os.getenv("BUCKET_NAME"))


# -- Wrapper function to generate the foodbot agent --
def generate_foodbot_agent(
    chat_store_token_limit=4096,
    verbose=True,
    callback: Optional[Literal["chainlit", "none"]] = "chainlit",
) -> ParseParamsAgent:
    """Initialize the agent with tools and settings.
    Args:
        chat_store_token_limit (int): The token limit for the chat memory.
        verbose (bool): Whether to enable verbose logging.
        callback (Optional[Literal[True, False]]): Whether to enable the Streamlit callback. Defaults to True.
    Returns:
        ParseParamsAgent: An instance of the ParseParamsAgent configured with the specified tools and settings.
    """
    if callback == "chainlit":
        callback_manager = CallbackManager([ChainlitStatusCallback()])
    else:
        callback_manager = None

    chat_store = SimpleChatStore()
    chat_memory = ChatMemoryBuffer.from_defaults(token_limit=chat_store_token_limit, chat_store=chat_store)

    try:
        agent_kwargs = {
            "tools": [
                candidate_generation_and_ranking_tool,
                enrich_restaurant_recommendations_tool,
            ],
            "llm": agent_llm_model,
            "memory": chat_memory,
            "callback_manager": callback_manager,
            "verbose": verbose,
        }
        return ParseParamsAgent.from_tools(**agent_kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize agent: {str(e)}")


# -- Function to initialize user session and preferences --
async def init_user_session():
    """
    1. Get authenticated user
    2. Initialize S3 client
    3. Load (or create) get_config_file() on S3
    4. Ensure user preferences exist
    5. Create the agent and store everything in user_session
    Returns: username, welcome message
    """
    # -- 1. Get authenticated user
    user = cl.user_session.get("user")
    username = user.identifier

    # -- 2. Load/initialize config file on S3
    try:
        objects = s3_client.list_objects()
        if not any(obj["file"] == get_config_file() for obj in objects):
            default_conf = {"credentials": {"usernames": {}}, "preferences": {}}
            s3_client.write_object(get_config_file(), yaml.dump(default_conf))

        raw = s3_client.read_object(get_config_file())
        config = yaml.safe_load(raw)
        cl.user_session.set("config", config)
    except Exception as e:
        await cl.Message(content=f"Error handling config file in S3: {e}").send()
        return None, None

    # -- 3. Set user and preferences
    cl.user_session.set("username", username)

    default_prefs = {
        "food_score": 0.55,
        "ambience_score": 0.15,
        "price_score": 0.15,
        "service_score": 0.15,
        "distance_preference": False,
        "distance_km": 15,
    }
    prefs_container = config.setdefault("preferences", {})
    user_prefs = prefs_container.get(username, default_prefs.copy())

    if username not in prefs_container:
        prefs_container[username] = user_prefs
        s3_client.write_object(get_config_file(), yaml.dump(config))

    cl.user_session.set("user_preferences", user_prefs)

    # -- 4. Agent instantiation
    try:
        agent = generate_foodbot_agent(chat_store_token_limit=4096, callback="chainlit", verbose=True)
        cl.user_session.set("agent", agent)
    except Exception as e:
        await cl.Message(content=f"Error initializing agent: {e}").send()
        return None, None

    # -- 5. Welcome message with user's name
    display_name = user.display_name or "food lover"
    wm = get_welcome_message(name=display_name)
    return username, wm


def get_chat_settings() -> cl.ChatSettings:
    """
    Generate chat settings for the user preferences.
    """
    prefs = cl.user_session.get("user_preferences", {})
    slider_score_settings = {"min": 0.0, "max": 1.0, "step": 0.05}

    return cl.ChatSettings(
        [
            cliw.Slider(id="food_score", label="🍽️ Food Score", initial=prefs["food_score"], **slider_score_settings),
            cliw.Slider(id="ambience_score", label="✨ Ambience Score", initial=prefs["ambience_score"], **slider_score_settings),
            cliw.Slider(id="price_score", label="💰 Price Score", initial=prefs["price_score"], **slider_score_settings),
            cliw.Slider(id="service_score", label="🤝 Service Score", initial=prefs["service_score"], **slider_score_settings),
            cliw.Switch(id="distance_preference", label="📍 Prefer Nearby Restaurants", initial=prefs.get("distance_preference", False)),
            cliw.Slider(id="max_distance", label="🚗 Max Distance (km)", initial=prefs.get("distance_km", 15), min=1, max=30, step=1),
        ]
    )
