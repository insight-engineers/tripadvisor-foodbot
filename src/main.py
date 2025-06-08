import os
from typing import Literal, Optional

from src.helper.utils import load_dotenv_file
from src.s3.client import S3Client

load_dotenv_file(dotenv_path=".env")

from llama_index.core.callbacks import CallbackManager
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore

from src.chat.agent import ParseParamsAgent
from src.chat.callbacks import ChainlitStatusCallback, StreamlitStatusCallback
from src.chat.client import agent_llm_model
from src.chat.tools import (
    candidate_generation_and_ranking_tool,
    enrich_restaurant_recommendations_tool,
)

s3_client = S3Client(bucket_name=os.getenv("AWS_BUCKET_NAME"))


# -- Wrapper function to generate the foodbot agent --
def generate_foodbot_agent(
    chat_store_token_limit=4096,
    verbose=True,
    callback: Optional[Literal["chainlit", "streamlit", "none"]] = "streamlit",
) -> ParseParamsAgent:
    """Initialize the agent with tools and settings.
    Args:
        chat_store_token_limit (int): The token limit for the chat memory.
        verbose (bool): Whether to enable verbose logging.
        callback (Optional[Literal[True, False]]): Whether to enable the Streamlit callback. Defaults to True.
    Returns:
        ParseParamsAgent: An instance of the ParseParamsAgent configured with the specified tools and settings.
    """
    if callback == "streamlit":
        callback_manager = CallbackManager([StreamlitStatusCallback()])
    elif callback == "chainlit":
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
