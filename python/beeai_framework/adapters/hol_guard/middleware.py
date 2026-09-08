# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from beeai_framework.context import RunContext, RunContextStartEvent, RunMiddlewareProtocol
from beeai_framework.emitter import EmitterOptions, EventMeta
from beeai_framework.emitter.utils import create_internal_event_matcher
from beeai_framework.tools import Tool
from beeai_framework.tools.types import StringToolOutput


class HOLGuardMiddleware(RunMiddlewareProtocol):
    """Evaluate BeeAI tool calls with a local HOL Guard process before execution.

    By default, ``fail_closed=True``: if the HOL Guard executable is unavailable,
    times out, or cannot be started, the tool call is blocked rather than executed.
    Set ``fail_closed=False`` only when explicit fail-open behavior is desired.
    """

    def __init__(
        self,
        *,
        executable: str = "hol-guard",
        timeout: float = 10.0,
        match_nested: bool = True,
        fail_closed: bool = True,
        workspace: str | Path | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        self.executable = executable
        self.timeout = timeout
        self.match_nested = match_nested
        self.fail_closed = fail_closed
        self.workspace = Path(workspace).expanduser() if workspace is not None else None
        self._cleanups: list[Callable[[], None]] = []

    def bind(self, ctx: RunContext) -> None:
        while self._cleanups:
            self._cleanups.pop()()

        @ctx.emitter.on(
            create_internal_event_matcher("start"),
            options=EmitterOptions(match_nested=self.match_nested, is_blocking=True, priority=100),
        )
        async def handle_start(data: Any, meta: EventMeta) -> None:
            if not isinstance(data, RunContextStartEvent) or not isinstance(meta.creator, RunContext):
                return
            creator = meta.creator
            if not isinstance(creator.instance, Tool):
                return

            tool = creator.instance
            payload = self._build_payload(tool=tool, data=data, meta=meta, ctx=creator)
            allowed, reason = await self._evaluate(payload)
            if allowed:
                return

            data.output = StringToolOutput(
                result=f"HOL Guard blocked BeeAI tool '{tool.name}' before execution: {reason}"
            )

        self._cleanups.append(lambda: ctx.emitter.off(callback=handle_start))

    def _build_payload(
        self,
        *,
        tool: Tool,
        data: RunContextStartEvent,
        meta: EventMeta,
        ctx: RunContext,
    ) -> dict[str, Any]:
        tool_input = data.input.get("input")
        if isinstance(tool_input, BaseModel):
            tool_input = tool_input.model_dump(mode="json")
        elif isinstance(tool_input, Mapping):
            tool_input = dict(tool_input)

        workspace = self.workspace or Path.cwd()
        run_id = meta.trace.run_id if meta.trace and meta.trace.run_id else ctx.run_id
        return {
            "artifact_id": f"beeai:tool:{tool.name}",
            "artifact_name": tool.name,
            "hook_event_name": "PreToolUse",
            "source_scope": "project",
            "tool_name": tool.name,
            "tool_input": tool_input,
            "cwd": str(workspace),
            "session_id": run_id,
        }

    async def _evaluate(self, payload: dict[str, Any]) -> tuple[bool, str]:
        try:
            process = await asyncio.create_subprocess_exec(
                self.executable,
                "hook",
                "--harness",
                "beeai",
                "--json",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace) if self.workspace is not None else None,
            )
        except OSError as error:
            return self._availability_result(f"HOL Guard could not start: {error}")

        encoded = json.dumps(payload, default=str, separators=(",", ":")).encode()
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(encoded), timeout=self.timeout)
        except TimeoutError:
            process.kill()
            await process.communicate()
            return self._availability_result(f"HOL Guard timed out after {self.timeout:g}s")

        if process.returncode == 0:
            return True, "allow"
        return False, self._decision_reason(stdout=stdout, stderr=stderr, returncode=process.returncode)

    def _availability_result(self, reason: str) -> tuple[bool, str]:
        return (not self.fail_closed, reason)

    @staticmethod
    def _decision_reason(*, stdout: bytes, stderr: bytes, returncode: int | None) -> str:
        output = stdout.decode(errors="replace").strip()
        if output:
            try:
                decision = json.loads(output)
            except json.JSONDecodeError:
                decision = None
            if isinstance(decision, dict):
                action = decision.get("policy_action")
                for key in ("permission_decision_reason", "reason", "message"):
                    value = decision.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
                if isinstance(action, str) and action.strip():
                    return f"policy action: {action.strip()}"

        error = stderr.decode(errors="replace").strip()
        if error:
            return error[:1000]
        return f"HOL Guard exited with status {returncode}"


__all__ = ["HOLGuardMiddleware"]
