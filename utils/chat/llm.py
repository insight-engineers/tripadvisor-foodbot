from llama_index.llms.openai import OpenAI as AgentOpenAI
from openai import OpenAI as CoreOpenAI
import os

llm_settings = {"model": "gpt-4o-mini", "temperature": 0.15}
agent_llm_model = AgentOpenAI(streaming=False, **llm_settings)
core_llm_model = CoreOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
