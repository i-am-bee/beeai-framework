# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
"""Per-turn context that installs ACP-routed backends for the duration of a prompt.

When the adapter is serving a turn, this context manager swaps the global
`ShellBackend`, `FileBackend`, and `io_confirm` handlers onto ACP-routed
implementations, so generic tools (`ShellTool`, `FileReadTool`, `FileEditTool`)
and the framework's `io_confirm` dispatch automatically — the tool code itself
stays protocol-agnostic. Mirrors the same ContextVar pattern used by
`beeai_framework/utils/io.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Self

from typing_extensions import Unpack

try:
    from acp.schema import AllowedOutcome, PermissionOption, ToolCallUpdate
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [acp_zed] not found.\nRun 'pip install \"beeai-framework[acp-zed]\"' to install."
    ) from e

from beeai_framework.adapters.acp_zed.serve.backends import ACPFileBackend, ACPShellBackend
from beeai_framework.tools.code import setup_shell_backend
from beeai_framework.tools.filesystem import setup_file_backend
from beeai_framework.utils.io import IOConfirmKwargs, setup_io_context

if TYPE_CHECKING:
    from beeai_framework.adapters.acp_zed.serve.agent import FsBridge


class ACPZedIOContext:
    """Installs ACP backends (shell + file + io_confirm) on enter, restores on exit.

    Any agent code that runs `ShellTool()`, `FileReadTool()`, `FileEditTool()`, or
    `io_confirm(...)` from inside this context is transparently routed through the
    ACP client — no ACP-specific tool subclasses required.
    """

    def __init__(self, bridge: FsBridge) -> None:
        self._bridge = bridge
        self._cleanups: list[Callable[[], None]] = []

    def __enter__(self) -> Self:
        self._cleanups = [
            setup_shell_backend(ACPShellBackend(self._bridge)),
            setup_file_backend(ACPFileBackend(self._bridge)),
            # pyrefly: ignore [bad-argument-type]
            setup_io_context(read=self._read, confirm=self._confirm),
        ]
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        for cleanup in reversed(self._cleanups):
            cleanup()
        self._cleanups = []

    async def _read(self, prompt: str) -> str:
        raise RuntimeError(
            "io_read is not supported under the ACPZed adapter — ACP has no free-form "
            "text-input method. Use a tool call or session/request_permission instead."
        )

    async def _confirm(self, prompt: str, **kwargs: Unpack[IOConfirmKwargs]) -> bool:
        title = kwargs.get("title", prompt)
        data = kwargs.get("data", {})
        submit_label = kwargs.get("submit_label", "Allow")
        cancel_label = kwargs.get("cancel_label", "Deny")

        tool_call = ToolCallUpdate(
            tool_call_id=f"confirm-{id(data)}",
            title=title,
            raw_input=data if isinstance(data, dict) else {"data": data},
            status="pending",
        )
        response = await self._bridge.request_permission(
            options=[
                PermissionOption(option_id="allow", name=submit_label, kind="allow_once"),
                PermissionOption(option_id="deny", name=cancel_label, kind="reject_once"),
            ],
            tool_call=tool_call,
        )
        return isinstance(response.outcome, AllowedOutcome) and response.outcome.option_id == "allow"
