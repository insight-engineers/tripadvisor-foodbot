import os

from dotenv import load_dotenv

if os.path.exists(".env"):
    load_dotenv(dotenv_path=".env")

import streamlit as st
from llama_index.core import Settings
from llama_index.core.agent import AgentChatResponse
from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks.base import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore

from utils.chat.agent import ParseParamsAgent
from utils.chat.client import agent_llm_model
from utils.chat.prompt import WELCOME_PROMPT
from utils.chat.tools import (
    candidate_generation_and_ranking_tool,
    enrich_restaurant_recommendations_tool,
)
from utils.helper import generate_streaming_response


# --- Streamlit Status Callback for display agent tools ---
class StreamlitStatusCallback(BaseCallbackHandler):
    def __init__(self):
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self._current_tool_name = None  # Cache for tool name

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: dict = None,
        event_id: str = None,
        **kwargs,
    ) -> None:
        if "status" in st.session_state:
            try:
                if event_type == CBEventType.AGENT_STEP:
                    st.session_state.status.update(label="Agent started processing 🔄", state="running")
                    st.session_state.progress_bar.progress(10)
                elif event_type == CBEventType.FUNCTION_CALL:
                    self._current_tool_name = (
                        payload.get(EventPayload.TOOL).name
                        if payload and EventPayload.TOOL in payload
                        else "Unknown tool"
                    )
                    st.session_state.status.update(
                        label=f"Calling tool: {self._current_tool_name} 🔄",
                        state="running",
                    )
            except Exception as e:
                st.session_state.status.update(label=f"Error in callback: {str(e)}", state="error")

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: dict = None,
        event_id: str = None,
        **kwargs,
    ) -> None:
        if "status" in st.session_state:
            try:
                if event_type == CBEventType.AGENT_STEP:
                    st.session_state.status.update(label="Agent has completed processing ✅", state="complete")
                    st.session_state.progress_bar.progress(90)
                elif event_type == CBEventType.FUNCTION_CALL:
                    tool_name = self._current_tool_name if self._current_tool_name else "Unknown tool"
                    st.session_state.status.update(label=f"Tool {tool_name} completed ✅", state="complete")
                    st.session_state.progress_bar.progress(60)
            except Exception as e:
                st.session_state.status.update(label=f"Error in callback: {str(e)}", state="error")
            finally:
                if event_type == CBEventType.FUNCTION_CALL:
                    self._current_tool_name = None  # Reset after use

    def start_trace(self, trace_id: str = None) -> None:
        pass

    def end_trace(self, trace_id: str = None, trace_map: dict = None) -> None:
        pass


# --- Load environment variables ---
assistant_avatar = "public/favicon.png"
user_avatar = "👤"

# --- Set page configuration ---
st.set_page_config(page_title="TripAdvisor Chatbot", page_icon=assistant_avatar, layout="centered")

# --- Initialize agent with callback ---
if "agent" not in st.session_state:
    callback = StreamlitStatusCallback()
    Settings.callback_manager = CallbackManager([callback])
    chat_store = SimpleChatStore()
    chat_memory = ChatMemoryBuffer.from_defaults(token_limit=4096, chat_store=chat_store)
    st.session_state.agent = ParseParamsAgent.from_tools(
        tools=[
            candidate_generation_and_ranking_tool,
            enrich_restaurant_recommendations_tool,
        ],
        llm=agent_llm_model,
        callback_manager=Settings.callback_manager,
        memory=chat_memory,
        verbose=True,
    )
    st.session_state.messages = []

# --- Sidebar for preferences ---
st.sidebar.image("public/banner.png", output_format="PNG", width=200)
with st.sidebar:
    st.markdown("### 🔧 Preference Tuning")
    st.session_state.distance_preference = st.checkbox(
        "Prefer nearby restaurants",
        value=st.session_state.get("distance_preference", False),
    )
    if st.session_state.distance_preference:
        st.session_state.distance_km = st.slider(
            "Maximum distance (km)", 1, 30, st.session_state.get("distance_km", 15), 1
        )
    st.session_state.user_preferences = {
        "food_score": st.slider("🍽️ Food", 0.0, 1.0, st.session_state.get("food_score", 0.55), 0.05),
        "ambience_score": st.slider("🏢 Ambience", 0.0, 1.0, st.session_state.get("ambience_score", 0.15), 0.05),
        "price_score": st.slider("💰 Price", 0.0, 1.0, st.session_state.get("price_score", 0.15), 0.05),
        "service_score": st.slider("👨‍🍳 Service", 0.0, 1.0, st.session_state.get("service_score", 0.15), 0.05),
    }
    st.session_state.update(st.session_state.user_preferences)

# --- Welcome message ---
with st.chat_message("assistant", avatar=assistant_avatar):
    st.markdown(WELCOME_PROMPT, unsafe_allow_html=True)

# --- Display chat history ---
for index, message in enumerate(st.session_state.messages):
    with st.chat_message(
        message["role"],
        avatar=assistant_avatar if message["role"] == "assistant" else user_avatar,
    ):
        st.markdown(message["content"])

# --- Handle user input ---
if prompt := st.chat_input("Planning your next meal? Ask me anything!"):
    if "progress_bar" in st.session_state:
        st.session_state.progress_bar.empty()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=user_avatar):
        st.markdown(prompt)

    st.session_state.status = st.status("Thinking...")
    progress_text = "Operation in progress. Please wait."
    st.session_state.progress_bar = st.progress(0)
    response: AgentChatResponse = st.session_state.agent.params_chat(
        prompt, **{"user_preferences": st.session_state.user_preferences}
    )
    st.session_state.progress_bar.progress(100)
    st.toast("Agent has ranked the restaurants and found the best for you", icon="✅")
    st.session_state.messages.append({"role": "assistant", "content": str(response)})

    with st.chat_message("assistant", avatar=assistant_avatar):
        st.write_stream(generate_streaming_response(str(response)))
