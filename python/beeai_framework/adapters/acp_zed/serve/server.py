# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from typing import Any, Generic, cast

try:
    from acp import (
        run_agent,
        text_block,
        update_agent_message,
        update_agent_thought,
    )
    from acp.interfaces import Client
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [acp_zed] not found.\nRun 'pip install \"beeai-framework[acp-zed]\"' to install."
    ) from e

from pydantic import BaseModel
from typing_extensions import TypeVar

from beeai_framework.adapters.acp_zed.serve._utils import acp_zed_prompt_to_framework_msgs
from beeai_framework.adapters.acp_zed.serve.agent import ACPZedServerAgent, FsBridge, PromptBlock
from beeai_framework.agents import AnyAgent
from beeai_framework.agents.react.agent import ReActAgent
from beeai_framework.agents.react.events import ReActAgentUpdateEvent
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.events import RequirementAgentSuccessEvent
from beeai_framework.agents.tool_calling.agent import ToolCallingAgent  # pyrefly: ignore [deprecated]
from beeai_framework.agents.tool_calling.events import ToolCallingAgentSuccessEvent
from beeai_framework.backend.message import AssistantMessage, ToolMessage
from beeai_framework.serve.errors import FactoryAlreadyRegisteredError
from beeai_framework.serve.server import Server
from beeai_framework.serve.utils import MemoryManager, init_agent_memory
from beeai_framework.utils import ModelLike
from beeai_framework.utils.cloneable import Cloneable
from beeai_framework.utils.lists import find_index
from beeai_framework.utils.models import to_model

AnyAgentLike = TypeVar("AnyAgentLike", bound=AnyAgent, default=AnyAgent)


class ACPZedServerConfig(BaseModel):
    """Configuration for `ACPZedServer`.

    Zed's ACP runs one agent per stdio process, so there is no port/host to configure.
    """

    agent_name: str | None = None
    agent_description: str | None = None


class ACPZedServer(Generic[AnyAgentLike], Server[AnyAgentLike, ACPZedServerAgent, ACPZedServerConfig]):
    """Expose a single BeeAI agent as a Zed Agent Client Protocol (ACP) program over stdio.

    Zed `settings.json`:

        {
            "agent_servers": {
                "beeai": {"command": "python", "args": ["/abs/path/to/entrypoint.py"]}
            }
        }
    """

    def __init__(
        self,
        *,
        config: ModelLike[ACPZedServerConfig] | None = None,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        super().__init__(config=to_model(ACPZedServerConfig, config or {}), memory_manager=memory_manager)
        self._bridge = FsBridge()

    @property
    def bridge(self) -> FsBridge:
        return self._bridge

    def serve(self) -> None:
        if len(self._members) != 1:
            raise ValueError(f"ACPZedServer exposes exactly one agent per process, got {len(self._members)}")
        member = self._members[0]
        factory = type(self)._factories[type(member)]  # type: ignore[index]
        wrapper: ACPZedServerAgent = factory(member, server=self)  # type: ignore[call-arg]
        _redirect_stdout_logging()
        asyncio.run(run_agent(wrapper))

    def _build_wrapper(self, agent: AnyAgent, run_turn: Any) -> ACPZedServerAgent:
        """Shared construction path used by all registered agent-type factories."""

        async def session_factory(session_id: str) -> AnyAgent:
            cloned = cast(AnyAgent, await agent.clone()) if isinstance(agent, Cloneable) else agent
            await init_agent_memory(cloned, self._memory_manager, session_id)
            return cloned

        return ACPZedServerAgent(  # pyrefly: ignore [bad-instantiation]
            name=self._config.agent_name or agent.meta.name,
            description=self._config.agent_description or agent.meta.description,
            bridge=self._bridge,
            session_factory=session_factory,
            run_turn=run_turn,
        )


def _redirect_stdout_logging() -> None:
    """Redirect any logging handler bound to stdout onto stderr.

    ACP's stdio transport uses stdout exclusively for JSON-RPC frames; a stray log
    line there corrupts the framing. Walks the root logger plus every named logger
    (including those with `propagate=False`).
    """
    loggers: list[logging.Logger] = [logging.root]
    loggers.extend(
        logger for logger in logging.Logger.manager.loggerDict.values() if isinstance(logger, logging.Logger)
    )
    for logger in loggers:
        for handler in logger.handlers:
            if getattr(handler, "stream", None) is sys.stdout:
                handler.stream = sys.stderr  # type: ignore[attr-defined]


async def _stream_messages_since(conn: Client, session_id: str, messages: list[Any], last_seen: Any | None) -> Any:
    """Emit `session/update` for every message appended since `last_seen`.

    Assistant text → `agent_message_chunk`. Tool-call requests on an assistant
    message → `agent_thought_chunk` (so the user sees the invocation even when the
    assistant sent no accompanying text). Tool results → `agent_thought_chunk` too.
    """
    start = find_index(messages, lambda m: m is last_seen, fallback=-1, reverse_traversal=True) + 1
    for msg in messages[start:]:
        last_seen = msg
        if isinstance(msg, ToolMessage):
            text = getattr(msg, "text", None) or str(msg)
            await conn.session_update(
                session_id=session_id, update=update_agent_thought(text_block(f"[tool result] {text}"))
            )
        elif isinstance(msg, AssistantMessage):
            text = getattr(msg, "text", None)
            if text:
                await conn.session_update(session_id=session_id, update=update_agent_message(text_block(text)))
            for call in msg.get_tool_calls():
                await conn.session_update(
                    session_id=session_id,
                    update=update_agent_thought(text_block(f"[tool call] {call.tool_name}({call.args})")),
                )
    return last_seen


def _requirement_agent_factory(agent: RequirementAgent, *, server: ACPZedServer[Any]) -> ACPZedServerAgent:
    async def run_turn(session_id: str, prompt: list[PromptBlock], conn: Client, session: RequirementAgent) -> str:
        last_seen: Any = None
        async for data, _event in session.run(acp_zed_prompt_to_framework_msgs(prompt)):
            last_seen = await _stream_messages_since(conn, session_id, data.state.memory.messages, last_seen)
            if isinstance(data, RequirementAgentSuccessEvent) and data.state.answer is not None:
                await conn.session_update(
                    session_id=session_id, update=update_agent_message(text_block(data.state.answer.text))
                )
        return "end_turn"

    return server._build_wrapper(agent, run_turn)


def _tool_calling_agent_factory(agent: ToolCallingAgent, *, server: ACPZedServer[Any]) -> ACPZedServerAgent:
    async def run_turn(session_id: str, prompt: list[PromptBlock], conn: Client, session: ToolCallingAgent) -> str:
        last_seen: Any = None
        async for data, _event in session.run(acp_zed_prompt_to_framework_msgs(prompt)):
            last_seen = await _stream_messages_since(conn, session_id, data.state.memory.messages, last_seen)
            if isinstance(data, ToolCallingAgentSuccessEvent) and data.state.result is not None:
                await conn.session_update(
                    session_id=session_id, update=update_agent_message(text_block(data.state.result.text))
                )
        return "end_turn"

    return server._build_wrapper(agent, run_turn)


def _react_agent_factory(agent: ReActAgent, *, server: ACPZedServer[Any]) -> ACPZedServerAgent:
    async def run_turn(session_id: str, prompt: list[PromptBlock], conn: Client, session: ReActAgent) -> str:
        async for data, event in session.run(acp_zed_prompt_to_framework_msgs(prompt)):
            if not isinstance(data, ReActAgentUpdateEvent) or event.name != "partial_update":
                continue
            value = data.update.value
            text = value if isinstance(value, str) else value.get_text_content()
            key = data.update.key
            if key == "final_answer":
                update = update_agent_message(text_block(text))
            elif key in ("thought", "tool_name", "tool_input", "tool_output"):
                update = update_agent_thought(text_block(f"[{key}] {text}"))
            else:
                continue
            await conn.session_update(session_id=session_id, update=update)
        return "end_turn"

    return server._build_wrapper(agent, run_turn)


with contextlib.suppress(FactoryAlreadyRegisteredError):
    ACPZedServer.register_factory(RequirementAgent, _requirement_agent_factory)  # type: ignore[arg-type]
with contextlib.suppress(FactoryAlreadyRegisteredError):
    ACPZedServer.register_factory(ToolCallingAgent, _tool_calling_agent_factory)  # type: ignore[arg-type]
with contextlib.suppress(FactoryAlreadyRegisteredError):
    ACPZedServer.register_factory(ReActAgent, _react_agent_factory)  # type: ignore[arg-type]
