# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import uuid
from typing import Any

from beeai_framework.agents import AgentError, AgentExecutionConfig
from beeai_framework.agents._utils import run_tools
from beeai_framework.agents.requirement.events import (
    RequirementAgentFinalAnswerEvent,
    RequirementAgentStartEvent,
    RequirementAgentSuccessEvent,
)
from beeai_framework.agents.requirement.prompts import RequirementAgentToolErrorPromptInput
from beeai_framework.agents.requirement.requirements import Requirement, Rule
from beeai_framework.agents.requirement.types import (
    RequirementAgentRequest,
    RequirementAgentRunState,
    RequirementAgentRunStateStep,
    RequirementAgentTemplates,
)
from beeai_framework.agents.requirement.utils._llm import RequirementsReasoner, _create_system_message
from beeai_framework.agents.requirement.utils._tool import FinalAnswerTool, FinalAnswerToolSchema
from beeai_framework.agents.tool_calling.utils import ToolCallChecker
from beeai_framework.backend import (
    AnyMessage,
    AssistantMessage,
    ChatModel,
    ChatModelOutput,
    MessageToolCallContent,
    MessageToolResultContent,
    ToolMessage,
)
from beeai_framework.backend.chat import ChatModelOptions
from beeai_framework.backend.errors import ChatModelToolCallError
from beeai_framework.backend.utils import parse_broken_json
from beeai_framework.context import RunContext
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.memory.utils import TEMP_MESSAGE_META_KEY, delete_messages_by_meta_key
from beeai_framework.middleware.stream_tool_call import StreamToolCallMiddleware
from beeai_framework.tools import AnyTool
from beeai_framework.utils.counter import RetryCounter
from beeai_framework.utils.lists import ensure_strictly_increasing, find_last_index
from beeai_framework.utils.strings import find_first_pair, generate_random_string, to_json


class RequirementAgentRunner:
    """Class responsible for running the agent."""

    def __init__(
        self,
        *,
        config: AgentExecutionConfig,
        tool_call_cycle_checker: ToolCallChecker,
        force_final_answer_as_tool: bool,
        expected_output: Any,
        run_context: RunContext,
        requirements: list[Requirement[RequirementAgentRunState]],
        tools: list[AnyTool],
        templates: RequirementAgentTemplates,
        llm: ChatModel,
    ) -> None:
        self._ctx = run_context
        self._llm = llm
        self._templates = templates
        self._force_final_answer_as_tool = force_final_answer_as_tool
        self._state = RequirementAgentRunState(
            answer=None, result=None, memory=UnconstrainedMemory(), steps=[], iteration=0
        )
        self._requirements = requirements
        self._reasoner = RequirementsReasoner(
            tools=tools,
            final_answer=FinalAnswerTool(expected_output, state=self._state),
            context=run_context,
        )
        self._run_config = config
        self._tool_call_cycle_checker = tool_call_cycle_checker

        max_retries_per_iteration = 0 if config.max_retries_per_step is None else config.max_retries_per_step
        self._iteration_error_counter = RetryCounter(error_type=AgentError, max_retries=max_retries_per_iteration)

        max_retries = 0 if config.total_max_retries is None else config.total_max_retries
        max_retries = max(max_retries_per_iteration, max_retries)
        self._global_error_counter = RetryCounter(error_type=AgentError, max_retries=max_retries)

    def _increment_iteration(self) -> None:
        self._state.iteration += 1

        if self._run_config.max_iterations and self._state.iteration > self._run_config.max_iterations:
            raise AgentError(f"Agent was not able to resolve the task in {self._state.iteration} iterations.")

    def __create_final_answer_stream(self, final_answer_tool: FinalAnswerTool) -> StreamToolCallMiddleware:
        stream_middleware = StreamToolCallMiddleware(
            final_answer_tool,
            "response",  # from the default schema
            match_nested=False,
            force_streaming=False,
        )
        stream_middleware.emitter.on(
            "update",
            lambda data, meta: self._ctx.emitter.emit(
                "final_answer",
                RequirementAgentFinalAnswerEvent(
                    state=self._state, output=data.output, delta=data.delta, output_structured=None
                ),
            ),
        )
        return stream_middleware

    async def _run_llm(
        self,
        request: RequirementAgentRequest,
    ) -> ChatModelOutput:
        stream_middleware = self.__create_final_answer_stream(request.final_answer)

        try:
            messages, options = self._prepare_llm_request(request)
            response = await self._llm.run(messages, **options).middleware(stream_middleware)

            self._state.usage.merge(response.usage)
            self._state.cost.merge(response.cost)

            return response
        except ChatModelToolCallError as e:
            generated_content = e.generated_content or (e.response.get_text_content() if e.response else "")
            if not generated_content:
                raise e

            response = ChatModelOutput.from_chunks([e.response] if e.response else [])
            response.output.clear()
            response.output.append(AssistantMessage(generated_content))
            return response
        finally:
            stream_middleware.unbind()

    def _prepare_llm_request(self, request: RequirementAgentRequest) -> tuple[list[AnyMessage], ChatModelOptions]:
        messages = [
            _create_system_message(
                template=self._templates.system,
                request=request,
            ),
            *self._state.memory.messages,
        ]

        options = ChatModelOptions(
            max_retries=self._run_config.max_retries_per_step,
            tools=request.allowed_tools,
            tool_choice=request.tool_choice,
            stream_partial_tool_calls=True,
            fallback_tool=request.final_answer if request.can_stop else None,
        )

        cache_control_injection_points = [
            {
                "location": "message",
                "index": 1 if self._requirements else 0,  # system prompt might be dynamic when requirements are set
            },
            {
                "location": "message",
                "index": find_last_index(
                    messages,
                    lambda msg: not msg.meta.get(TEMP_MESSAGE_META_KEY)
                    # TODO: remove once https://github.com/BerriAI/litellm/issues/17479 is resolved
                    and (self._llm.provider_id != "amazon_bedrock" or not isinstance(msg, ToolMessage)),
                ),
            },
        ]
        options["cache_control_injection_points"] = ensure_strictly_increasing(  # type: ignore
            cache_control_injection_points,
            key=lambda v: v["index"],  # prevent duplicates
        )
        return messages, options

    async def _create_final_answer_tool_call(self, full_text: str) -> AssistantMessage | None:
        """Try to convert a text message to a valid final answer tool call."""

        json_object_pair = find_first_pair(full_text, ("{", "}"))
        final_answer_input = parse_broken_json(json_object_pair.outer) if json_object_pair else None
        if not final_answer_input and not self._reasoner.final_answer.custom_schema:
            final_answer_input = FinalAnswerToolSchema(response=full_text).model_dump()

        if not final_answer_input:
            return None

        manual_assistant_tool_call_message = MessageToolCallContent(
            type="tool-call",
            id=f"call_{generate_random_string(8).lower()}",
            tool_name=self._reasoner.final_answer.name,
            args=to_json(final_answer_input, sort_keys=False),
        )
        return AssistantMessage(manual_assistant_tool_call_message)

    async def _create_request(self, *, extra_rules: list[Rule] | None = None) -> RequirementAgentRequest:
        return await self._reasoner.create_request(
            self._state,
            force_tool_call=self._force_final_answer_as_tool,
            extra_rules=extra_rules,
        )

    async def _invoke_tool_calls(
        self, tools: list[AnyTool], tool_calls: list[MessageToolCallContent]
    ) -> list[ToolMessage]:
        tool_results: list[ToolMessage] = []

        for tool_call in await run_tools(
            tools=tools,
            messages=tool_calls,
            context={"state": self._state.model_dump()},
        ):
            self._state.steps.append(
                RequirementAgentRunStateStep(
                    id=str(uuid.uuid4()),
                    iteration=self._state.iteration,
                    input=tool_call.input,
                    output=tool_call.output,
                    tool=tool_call.tool,
                    error=tool_call.error,
                )
            )

            if tool_call.error is not None:
                result = self._templates.tool_error.render(
                    RequirementAgentToolErrorPromptInput(reason=tool_call.error.explain())
                )
            else:
                result = (
                    tool_call.output.get_text_content()
                    if not tool_call.output.is_empty()
                    else self._templates.tool_no_result.render(tool_call=tool_call)
                )

            tool_results.append(
                ToolMessage(
                    MessageToolResultContent(
                        tool_name=tool_call.tool.name if tool_call.tool else tool_call.msg.tool_name,
                        tool_call_id=tool_call.msg.id,
                        result=result,
                    )
                )
            )
            if tool_call.error is not None:
                self._iteration_error_counter.use(tool_call.error)
                self._global_error_counter.use(tool_call.error)

        return tool_results

    async def add_messages(self, messages: list[AnyMessage]) -> None:
        await self._state.memory.add_many(messages)

    async def run(self) -> RequirementAgentRunState:
        """Run the agent until it reaches the final answer. Returns the final state."""

        if self._state.answer is not None:
            return self._state

        # Init requirements
        await self._reasoner.update(self._requirements)

        while self._state.answer is None:
            self._increment_iteration()

            request = await self._create_request()
            await self._ctx.emitter.emit(
                "start",
                RequirementAgentStartEvent(state=self._state, request=request),
            )
            self._iteration_error_counter.reset()
            response = await self._run(request)
            await self._ctx.emitter.emit(
                "success",
                RequirementAgentSuccessEvent(state=self._state, response=response),
            )
        return self._state

    async def _run(self, request: RequirementAgentRequest) -> ChatModelOutput:
        """Run a single iteration of the agent."""

        response = await self._run_llm(request)

        # Try to cast a text message to a final answer tool call if it is allowed
        if not response.get_tool_calls():
            text = response.get_text_content()
            final_answer_tool_call = (
                await self._create_final_answer_tool_call(text) if request.can_stop and text else None
            )
            if final_answer_tool_call:
                # Manually emit the final_answer event
                stream = self.__create_final_answer_stream(request.final_answer)
                await stream.add(ChatModelOutput(output=[final_answer_tool_call]))
            else:
                err = AgentError("Model produced an invalid final answer tool call.")
                self._iteration_error_counter.use(err)
                self._global_error_counter.use(err)

                await self._reasoner.update(requirements=[])
                updated_request = await self._create_request(
                    extra_rules=[Rule(target=self._reasoner.final_answer.name, allowed=True, hidden=False)],
                )
                self._force_final_answer_as_tool = True
                return await self._run(updated_request)

            response.output_structured = None
            response.output = [final_answer_tool_call]

        # Check for cycles
        tool_calls = response.get_tool_calls()
        for tool_call_msg in tool_calls:
            self._tool_call_cycle_checker.register(tool_call_msg)
            if self._tool_call_cycle_checker.cycle_found:
                self._tool_call_cycle_checker.reset()
                updated_request = await self._create_request(
                    extra_rules=[Rule(target=tool_call_msg.tool_name, allowed=False, hidden=False, forced=True)],
                )
                return await self._run(updated_request)

        tool_results = await self._invoke_tool_calls(request.allowed_tools, tool_calls)

        await self._state.memory.add_many([*response.output, *tool_results])
        await delete_messages_by_meta_key(self._state.memory, key=TEMP_MESSAGE_META_KEY, value=True)

        return response
