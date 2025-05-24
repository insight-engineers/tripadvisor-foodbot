import json
import uuid
from typing import Any, List, Optional, Sequence, Union

import llama_index.core.instrumentation as instrument
from llama_index.core.agent import (
    AgentChatResponse,
    FunctionCallingAgent,
    FunctionCallingAgentWorker,
)
from llama_index.core.agent.function_calling.step import (
    DEFAULT_MAX_FUNCTION_CALLS,
    build_missing_tool_output,
    get_function_by_name,
)
from llama_index.core.agent.runner.base import AgentState
from llama_index.core.agent.types import Task, TaskStep, TaskStepOutput
from llama_index.core.agent.utils import add_user_step_to_memory
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.callbacks import (
    CallbackManager,
    CBEventType,
    EventPayload,
    trace_method,
)
from llama_index.core.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
    ChatResponseMode,
)
from llama_index.core.instrumentation.events.agent import (
    AgentChatWithStepEndEvent,
    AgentChatWithStepStartEvent,
    AgentRunStepEndEvent,
    AgentRunStepStartEvent,
    AgentToolCallEvent,
)
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.memory import BaseMemory
from llama_index.core.objects.base import ObjectRetriever
from llama_index.core.settings import Settings
from llama_index.core.tools import BaseTool, ToolOutput
from llama_index.core.tools.calling import call_tool
from llama_index.core.tools.types import ToolMetadata
from loguru import logger as log

dispatcher = instrument.get_dispatcher(__name__)


class ParseParamsAgent(FunctionCallingAgent):
    """Override chat method to parse and forward custom parameters to tool calls."""

    @classmethod
    def from_tools(
        cls,
        tools: Optional[List[BaseTool]] = None,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        llm: Optional[FunctionCallingLLM] = None,
        verbose: bool = False,
        max_function_calls: int = DEFAULT_MAX_FUNCTION_CALLS,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        prefix_messages: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        state: Optional[AgentState] = None,
        allow_parallel_tool_calls: bool = True,
        **additional_kwargs: Any,
    ) -> "FunctionCallingAgent":
        """Create a FunctionCallingAgent from a list of tools."""
        tools = tools or []

        llm = llm or Settings.llm  # type: ignore
        assert isinstance(llm, FunctionCallingLLM), "llm must be an instance of FunctionCallingLLM"

        if callback_manager is not None:
            llm.callback_manager = callback_manager

        if system_prompt is not None:
            if prefix_messages is not None:
                raise ValueError("Cannot specify both system_prompt and prefix_messages")
            prefix_messages = [ChatMessage(content=system_prompt, role="system")]

        prefix_messages = prefix_messages or []

        agent_worker = ParseParamsAgentWorker.from_tools(
            tools,
            tool_retriever=tool_retriever,
            llm=llm,
            verbose=verbose,
            max_function_calls=max_function_calls,
            callback_manager=callback_manager,
            prefix_messages=prefix_messages,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
        )

        return cls(
            agent_worker=agent_worker,
            memory=memory,
            chat_history=chat_history,
            state=state,
            llm=llm,
            callback_manager=callback_manager,
            verbose=verbose,
            **additional_kwargs,
        )

    def params_chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        **additional_kwargs,
    ) -> AgentChatResponse:
        if tool_choice is None:
            tool_choice = self.default_tool_choice
        with self.callback_manager.event(
            CBEventType.AGENT_STEP,
            payload={EventPayload.MESSAGES: [message]},
        ) as e:
            chat_response = self._chat(
                message=message,
                chat_history=chat_history,
                tool_choice=tool_choice,
                mode=ChatResponseMode.WAIT,
                **additional_kwargs,
            )
            assert isinstance(chat_response, AgentChatResponse)
            e.on_end(payload={EventPayload.RESPONSE: chat_response})
        return chat_response

    @dispatcher.span
    def _chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
        mode: ChatResponseMode = ChatResponseMode.WAIT,
        **additional_kwargs,
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        if chat_history is not None:
            self.memory.set(chat_history)
        task = self.create_task(message)

        result_output = None
        dispatcher.event(AgentChatWithStepStartEvent(user_msg=message))
        while True:
            cur_step_output = self._run_step(task.task_id, mode=mode, tool_choice=tool_choice, **additional_kwargs)

            if cur_step_output.is_last:
                result_output = cur_step_output
                break

            tool_choice = "auto"

        result = self.finalize_response(
            task.task_id,
            result_output,
        )
        dispatcher.event(AgentChatWithStepEndEvent(response=result))
        return result

    @dispatcher.span
    def _run_step(
        self,
        task_id: str,
        step: Optional[TaskStep] = None,
        input: Optional[str] = None,
        mode: ChatResponseMode = ChatResponseMode.WAIT,
        **additional_kwargs,
    ) -> TaskStepOutput:
        """Execute step."""
        task = self.state.get_task(task_id)
        step_queue = self.state.get_step_queue(task_id)
        step = step or step_queue.popleft()
        if input is not None:
            step.input = input

        dispatcher.event(AgentRunStepStartEvent(task_id=task_id, step=step, input=input))

        if self.verbose:
            log.info(f"> Running step {step.step_id}. Step input: {step.input}")

        # TODO: figure out if you can dynamically swap in different step executors
        # not clear when you would do that by theoretically possible

        if mode == ChatResponseMode.WAIT:
            cur_step_output = self.agent_worker.run_step(step, task, **additional_kwargs)
        elif mode == ChatResponseMode.STREAM:
            cur_step_output = self.agent_worker.stream_step(step, task, **additional_kwargs)

        # append cur_step_output next steps to queue
        next_steps = cur_step_output.next_steps
        step_queue.extend(next_steps)

        # add cur_step_output to completed steps
        completed_steps = self.state.get_completed_steps(task_id)
        completed_steps.append(cur_step_output)

        dispatcher.event(AgentRunStepEndEvent(step_output=cur_step_output))
        return cur_step_output


class ParseParamsAgentWorker(FunctionCallingAgentWorker):
    @trace_method("run_step")
    def run_step(self, step: TaskStep, task: Task, **additional_kwargs) -> TaskStepOutput:
        """Run step."""
        if step.input is not None:
            add_user_step_to_memory(step, task.extra_state["new_memory"], verbose=self._verbose)
        # TODO: see if we want to do step-based inputs
        tools = self.get_tools(task.input)

        # get response and tool call (if exists)
        response = self._llm.chat_with_tools(
            tools=tools,
            user_msg=None,
            chat_history=self.get_all_messages(task),
            verbose=self._verbose,
            allow_parallel_tool_calls=self.allow_parallel_tool_calls,
        )
        tool_calls = self._llm.get_tool_calls_from_response(response, error_on_no_tool_call=False)
        tool_outputs: List[ToolOutput] = []

        if self._verbose and response.message.content:
            log.info("=== LLM Response ===")
            log.info(str(response.message.content))

        if not self.allow_parallel_tool_calls and len(tool_calls) > 1:
            raise ValueError("Parallel tool calls not supported for synchronous function calling agent")

        # call all tools, gather responses
        task.extra_state["new_memory"].put(response.message)
        if len(tool_calls) == 0 or task.extra_state["n_function_calls"] >= self._max_function_calls:
            # we are done
            is_done = True
            new_steps = []
        else:
            is_done = False
            for i, tool_call in enumerate(tool_calls):
                # TODO: maybe execute this with multi-threading
                return_direct = self._call_function(
                    tools,
                    tool_call,
                    task.extra_state["new_memory"],
                    tool_outputs,
                    verbose=self._verbose,
                    **additional_kwargs,
                )
                task.extra_state["sources"].append(tool_outputs[-1])
                task.extra_state["n_function_calls"] += 1

                # check if any of the tools return directly -- only works if there is one tool call
                if i == 0 and return_direct:
                    is_done = True
                    response = task.extra_state["sources"][-1].content
                    break

            # put tool output in sources and memory
            new_steps = (
                [
                    step.get_next_step(
                        step_id=str(uuid.uuid4()),
                        # NOTE: input is unused
                        input=None,
                    )
                ]
                if not is_done
                else []
            )

        # get response string
        # return_direct can change the response type
        try:
            response_str = str(response.message.content)
        except AttributeError:
            response_str = str(response)

        agent_response = AgentChatResponse(response=response_str, sources=tool_outputs)

        return TaskStepOutput(
            output=agent_response,
            task_step=step,
            is_last=is_done,
            next_steps=new_steps,
        )

    def _call_function(
        self,
        tools: Sequence[BaseTool],
        tool_call: ToolSelection,
        memory: BaseMemory,
        sources: List[ToolOutput],
        verbose: bool = False,
        **additional_kwargs,
    ) -> bool:
        tool = get_function_by_name(tools, tool_call.tool_name)
        tool_args_str = json.dumps(tool_call.tool_kwargs)
        tool_metadata = tool.metadata if tool is not None else ToolMetadata(description="", name=tool_call.tool_name)

        dispatcher.event(AgentToolCallEvent(arguments=tool_args_str, tool=tool_metadata))
        with self.callback_manager.event(
            CBEventType.FUNCTION_CALL,
            payload={
                EventPayload.FUNCTION_CALL: tool_args_str,
                EventPayload.TOOL: tool_metadata,
            },
        ) as event:
            tool_output = (
                self.call_tool_with_additional_kwargs(tool_call, tools, verbose=verbose, **additional_kwargs)
                if tool is not None
                else build_missing_tool_output(tool_call)
            )
            event.on_end(payload={EventPayload.FUNCTION_OUTPUT: str(tool_output)})

        function_message = ChatMessage(
            content=str(tool_output),
            role=MessageRole.TOOL,
            additional_kwargs={
                "name": tool_call.tool_name,
                "tool_call_id": tool_call.tool_id,
            },
        )
        sources.append(tool_output)
        memory.put(function_message)

        return tool.metadata.return_direct if tool is not None else False

    def call_tool(self, tool: BaseTool, arguments: dict) -> ToolOutput:
        """Call a tool with arguments."""
        try:
            # merge the two dicts and update the arguments
            tool_kwargs = {
                **tool.metadata.get_parameters_dict()["properties"],
                **arguments,
            }
            return tool(**tool_kwargs)
        except Exception as e:
            return ToolOutput(
                content="Encountered error: " + str(e),
                tool_name=tool.metadata.name,
                raw_input=arguments,
                raw_output=str(e),
                is_error=True,
            )

    def call_tool_with_additional_kwargs(
        self,
        tool_call: ToolSelection,
        tools: Sequence["BaseTool"],
        verbose: bool = False,
        **additional_kwargs,
    ) -> ToolOutput:
        tools_by_name = {tool.metadata.name: tool for tool in tools}
        name = tool_call.tool_name
        tool_call.tool_kwargs.update(kwargs_dict=additional_kwargs)
        if verbose:
            arguments_str = json.dumps(tool_call.tool_kwargs)
            log.info("=== Calling Function ===")
            log.info(f"Calling function: {name} with args: {arguments_str}")
        tool = tools_by_name[name]
        output = self.call_tool(tool, tool_call.tool_kwargs)

        if verbose:
            log.info("=== Function Output ===")
            log.info(output.content)

        return output
