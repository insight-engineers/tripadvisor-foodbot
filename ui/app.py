import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chainlit as cl
import openai
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.agent import AgentChatResponse, ReActAgent
from llama_index.core.callbacks import CallbackManager
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore

from utils.chat.client import agent_llm_model
from utils.chat.tools import (
    candidate_generation_and_ranking_tool,
    enrich_restaurant_recommendations_tool,
)

try:
    load_dotenv(dotenv_path=".env", verbose=True)
    openai.api_key = os.environ.get("OPENAI_API_KEY")
except:
    raise Exception("Please provide an OpenAI API key in a .env file.")


@cl.on_chat_start
async def start():
    Settings.callback_manager = CallbackManager([cl.LlamaIndexCallbackHandler()])

    chat_store = SimpleChatStore()
    chat_memory = ChatMemoryBuffer.from_defaults(
        token_limit=3000,
        chat_store=chat_store,
        chat_store_key="user1",
    )

    agent = ReActAgent.from_tools(
        tools=[
            candidate_generation_and_ranking_tool,
            enrich_restaurant_recommendations_tool,
        ],
        llm=agent_llm_model,
        callback_manager=Settings.callback_manager,
        memory=chat_memory,
        verbose=True,
    )

    cl.user_session.set("agent", agent)  # Store the agent in session

    await cl.Message(
        author="Assistant",
        content=(
            '<div style="color: #00af87; font-weight: bold;">Welcome to the TripAdvisor Chatbot! 🎉</div>\n\n'
            "Discover the best dining spots in <b>Ho Chi Minh City</b> and <b>Hanoi</b>! 🍽️\n\n"
            '<div style="color: #00af87; font-weight: bold; margin-top: 1em;">What Can I Do for You? 🤔</div>\n\n'
            "- 🍽️ **Find Restaurants:** Looking for the perfect spot to eat? Let me help you discover hidden gems and popular favorites!\n\n"
            "- ⭐ **Get Recommendations:** Not sure where to go? I'll suggest top-rated places based on your preferences.\n\n"
            "🍲🍹 Bon appétit and happy exploring! 🥳🎉\n\n"
        ),
    ).send()

    cl.user_session.set("message_history", [])


@cl.on_message
async def main(message: cl.Message):
    agent: ReActAgent = cl.user_session.get("agent")  # type: ReActAgent
    msg: cl.Message = cl.Message(content="", author="Assistant")
    stream: AgentChatResponse = await cl.make_async(agent.stream_chat)(message.content)

    for part in stream.response_gen:
        await msg.stream_token(part)

    await msg.update()
