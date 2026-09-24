import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from beeai_framework.context import RunContext, RunContextStartEvent, RunMiddlewareProtocol
from beeai_framework.emitter import EmitterOptions, EventMeta
from beeai_framework.emitter.utils import create_internal_event_matcher
from beeai_framework.tools import Tool, tool
from beeai_framework.tools.types import StringToolOutput


class HOLGuardMiddleware(RunMiddlewareProtocol):
    """Example pre-tool middleware backed by the local HOL Guard runtime.

    Install HOL Guard separately, for example with ``pipx install hol-guard``.
    This example fails closed if Guard is unavailable, times out, crashes, or
    returns a non-allow decision.
    """

    def __init__(
        self,
        *,
        executable: str = "hol-guard",
        timeout: float = 10.0,
        workspace: str | Path | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        self.executable = executable
        self.timeout = timeout
        self.workspace = Path(workspace).expanduser() if workspace is not None else None
        self._bound_run_ids: set[str] = set()

    def bind(self, ctx: RunContext) -> None:
        if ctx.run_id in self._bound_run_ids:
            return
        self._bound_run_ids.add(ctx.run_id)

        @ctx.emitter.on(
            create_internal_event_matcher("start"),
            options=EmitterOptions(match_nested=True, is_blocking=True, priority=100),
        )
        async def handle_start(data: Any, meta: EventMeta) -> None:
            if not isinstance(data, RunContextStartEvent) or not isinstance(meta.creator, RunContext):
                return
            creator = meta.creator
            if not isinstance(creator.instance, Tool):
                return

            tool_instance = creator.instance
            payload = self._build_payload(tool=tool_instance, data=data, ctx=creator)
            allowed, reason = await self._evaluate(payload)
            if allowed:
                return

            data.output = StringToolOutput(
                result=f"HOL Guard blocked BeeAI tool '{tool_instance.name}' before execution: {reason}"
            )

        @ctx.emitter.on(
            create_internal_event_matcher("finish"),
            options=EmitterOptions(match_nested=False),
        )
        async def handle_finish(_: Any, __: EventMeta) -> None:
            ctx.emitter.off(callback=handle_start)
            ctx.emitter.off(callback=handle_finish)
            self._bound_run_ids.discard(ctx.run_id)

    def _build_payload(
        self,
        *,
        tool: Tool,
        data: RunContextStartEvent,
        ctx: RunContext,
    ) -> dict[str, Any]:
        tool_input = data.input.get("input")
        if isinstance(tool_input, BaseModel):
            tool_input = tool_input.model_dump(mode="json")
        elif isinstance(tool_input, Mapping):
            tool_input = dict(tool_input)

        workspace = self.workspace or Path.cwd()
        return {
            "artifact_id": f"beeai:tool:{tool.name}",
            "artifact_name": tool.name,
            "hook_event_name": "PreToolUse",
            "source_scope": "project",
            "tool_name": tool.name,
            "tool_input": tool_input,
            "cwd": str(workspace),
            "session_id": ctx.run_id,
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
            return False, f"HOL Guard could not start: {error}"

        encoded = json.dumps(payload, default=str, separators=(",", ":")).encode()
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(encoded), timeout=self.timeout)
        except TimeoutError:
            process.kill()
            await process.communicate()
            return False, f"HOL Guard timed out after {self.timeout:g}s"

        if process.returncode == 0:
            return True, "allow"

        return False, self._decision_reason(stdout=stdout, stderr=stderr, returncode=process.returncode)

    @staticmethod
    def _decision_reason(*, stdout: bytes, stderr: bytes, returncode: int | None) -> str:
        output = stdout.decode(errors="replace").strip()
        if output:
            try:
                decision = json.loads(output)
            except json.JSONDecodeError:
                decision = None
            if isinstance(decision, dict):
                for key in ("permission_decision_reason", "reason", "message"):
                    value = decision.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
                for key in ("policy_action", "minimum_action", "decision"):
                    value = decision.get(key)
                    if isinstance(value, str) and value.strip():
                        return f"policy action: {value.strip()}"

        error = stderr.decode(errors="replace").strip()
        if error:
            return error[:1000]
        return f"HOL Guard exited with status {returncode}"


@tool(name="Bash", description="Example shell tool protected by HOL Guard")
async def bash(command: str) -> str:
    return f"Tool would execute: {command}"


async def main() -> None:
    result = await bash.run({"command": "echo hello"}).middleware(HOLGuardMiddleware())
    print(result.get_text_content())


if __name__ == "__main__":
    asyncio.run(main())
