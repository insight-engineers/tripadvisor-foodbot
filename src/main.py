from src.helper.utils import load_dotenv_file

load_dotenv_file(dotenv_path=".env")


from llama_index.core.callbacks import CallbackManager
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore

from src.chat.agent import ParseParamsAgent
from src.chat.callback import StreamlitStatusCallback
from src.chat.client import agent_llm_model
from src.chat.tools import (
    candidate_generation_and_ranking_tool,
    enrich_restaurant_recommendations_tool,
)


# -- Wrapper function to generate the foodbot agent --
def generate_foodbot_agent(token_limit=4096, verbose=True, enable_streamlit_callback=True):
    """Initialize the agent with tools and settings."""
    callback_manager = CallbackManager([StreamlitStatusCallback()]) if enable_streamlit_callback else None
    chat_store = SimpleChatStore()
    chat_memory = ChatMemoryBuffer.from_defaults(token_limit=token_limit, chat_store=chat_store)

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
