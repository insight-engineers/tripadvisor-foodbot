import streamlit as st
from llama_index.core.callbacks.base import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload


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
                    st.session_state.progress_bar.progress(35)
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
                    st.session_state.progress_bar.progress(85)
                elif event_type == CBEventType.FUNCTION_CALL:
                    tool_name = self._current_tool_name if self._current_tool_name else "Unknown tool"
                    st.session_state.status.update(label=f"Tool {tool_name} completed ✅", state="complete")
                    st.session_state.progress_bar.progress(65)
            except Exception as e:
                st.session_state.status.update(label=f"Error in callback: {str(e)}", state="error")
            finally:
                if event_type == CBEventType.FUNCTION_CALL:
                    self._current_tool_name = None  # Reset after use

    def start_trace(self, trace_id: str = None) -> None:
        pass

    def end_trace(self, trace_id: str = None, trace_map: dict = None) -> None:
        pass
