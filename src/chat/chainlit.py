from typing import List, Union, cast

from chainlit import LlamaIndexCallbackHandler
from chainlit.action import Action
from chainlit.config import config
from chainlit.context import context
from chainlit.message import AskMessageBase
from chainlit.telemetry import trace_event
from chainlit.types import (
    AskActionResponse,
    AskActionSpec,
)
from literalai.helper import utc_now


class ChainlitStatusCallback(LlamaIndexCallbackHandler):
    """Chainlit callback handler for LlamaIndex events."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # No additional initialization needed
        # This class is used to integrate with Chainlit's callback system
        # NOTE: Custom if needed
        pass


class AskActionMessage(AskMessageBase):
    """
    Like the original AskActionMessage, but removes the **Selected action** text
    """

    def __init__(
        self,
        content: str,
        actions: List[Action],
        author=config.ui.name,
        timeout=1200,
        raise_on_timeout=False,
    ):
        self.content = content
        self.actions = actions
        self.author = author
        self.timeout = timeout
        self.raise_on_timeout = raise_on_timeout

        super().__post_init__()

    async def send(self) -> Union[AskActionResponse, None]:
        """
        Sends the question to ask to the UI and waits for the reply
        """
        trace_event("send_ask_action")

        if not self.created_at:
            self.created_at = utc_now()

        if not self.content:
            self.content = "\n"

        if self.streaming:
            self.streaming = False

        if config.code.author_rename:
            self.author = await config.code.author_rename(self.author)

        self.wait_for_answer = True

        step_dict = await self._create()

        action_keys = []

        for action in self.actions:
            action_keys.append(action.id)
            await action.send(for_id=str(step_dict["id"]))

        spec = AskActionSpec(
            type="action",
            step_id=step_dict["id"],
            timeout=self.timeout,
            keys=action_keys,
        )

        res = cast(
            Union[AskActionResponse, None],
            await context.emitter.send_ask_user(step_dict, spec, self.raise_on_timeout),
        )

        for action in self.actions:
            await action.remove()

        self.wait_for_answer = False

        await self.update()
        return res
