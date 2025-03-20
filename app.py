# Template for creating a simple ReAct agent using the ChainLit framework.
# This agent can perform simple mathematical operations like addition and multiplication.

import os
import openai
import chainlit as cl
from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

try:
    load_dotenv(dotenv_path=".env", verbose=True)
    openai.api_key = os.environ.get("OPENAI_API_KEY")
except:
    raise Exception("Please provide an OpenAI API key in a .env file.")

def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b

def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b

multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)

@cl.on_chat_start
async def start():
    llm_model = OpenAI(model="gpt-4o-mini", temperature=0.1, streaming=False)
    
    Settings.llm = llm_model
    Settings.context_window = 4096

    agent = ReActAgent.from_tools([multiply_tool, add_tool], llm=llm_model, verbose=True)
    cl.user_session.set("agent", agent)  # Store the agent in session

    await cl.Message(
        author="Assistant", content="Hello, sir! I'm here to help you with your mathematical queries."
    ).send()

@cl.on_message
async def main(message: cl.Message):
    agent: ReActAgent = cl.user_session.get("agent")  # type: ReActAgent

    response = agent.chat(message.content)

    await cl.Message(author="Assistant", content=response).send()
