from chainlit import LlamaIndexCallbackHandler


class ChainlitStatusCallback(LlamaIndexCallbackHandler):
    """Chainlit callback handler for LlamaIndex events."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # No additional initialization needed
        # This class is used to integrate with Chainlit's callback system
        # NOTE: Custom if needed
        pass
