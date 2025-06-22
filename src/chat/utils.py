import re
import time
from typing import Any, Dict, List

from src.chat.client import async_core_llm_model
from src.chat.models import NextResponse
from src.chat.prompt import CONV_SUMMARY_PROMPT, RESPONSE_SUGGESTION_PROMPT


async def generate_conv_summary(chat_history: List[Dict[str, str]]) -> str:
    """
    Generate a summary of the conversation messages using the async LLM model.
    This function formats the conversation messages into a string and sends it to the LLM model
    to generate a summary. The summary is streamed back in parts.
    """
    llm_params = {
        "messages": [
            {
                "role": "developer",
                "content": CONV_SUMMARY_PROMPT.format(chat_history=chat_history),
            }
        ],
        "model": "gpt-4.1-nano",
        "max_tokens": 512,
        "temperature": 0.6,
        "top_p": 1,
    }

    completion = await async_core_llm_model.chat.completions.create(**llm_params)

    return completion.choices[0].message.content


async def generate_next_response(chat_history: List[Dict[str, str]]) -> NextResponse:
    """
    Generate a list of 3 suggested questions based on the chat history.
    This function formats the chat history into a string and sends it to the LLM model
    to generate suggestions. The suggestions are streamed back in parts.
    """
    llm_params = {
        "messages": [
            {
                "role": "developer",
                "content": RESPONSE_SUGGESTION_PROMPT.format(chat_history=chat_history),
            }
        ],
        "model": "gpt-4.1-nano",
        "max_tokens": 512,
        "temperature": 1.0,
        "top_p": 1,
        "response_format": NextResponse,
    }

    async_completion = await async_core_llm_model.beta.chat.completions.parse(**llm_params)
    unable_response = ["I'm not sure what to ask next. Can you help me?"]
    completion_response = unable_response

    if async_completion.choices[0].message.parsed:
        completion_response = async_completion.choices[0].message.parsed.model_dump()

    return NextResponse(**completion_response).model_dump()


def generate_streaming_response(response: Any, delay: float = 0.05):
    """Generator to mimic some LLM response with spaces and newlines preserved"""
    for part in re.findall(r"\S+\s*", str(response)):
        yield part
        time.sleep(delay)
