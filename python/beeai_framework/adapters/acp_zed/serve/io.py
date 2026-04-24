# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

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

from beeai_framework.utils.io import IOConfirmKwargs, setup_io_context

if TYPE_CHECKING:
    from beeai_framework.adapters.acp_zed.serve.agent import FsBridge


class ACPZedIOContext:
    """Route `io_confirm` through ACP `session/request_permission` for the life of a turn.

    Mirrors the A2A / IBM-ACP `setup_io_context` pattern: when this context is active,
    any agent code calling `io_confirm(...)` (e.g. `AskPermissionRequirement`) gets the
    prompt delivered to the editor user via `session/request_permission` instead of
    stdin, and the user's choice flows back as a `bool`.

    `io_read` is overridden to raise — ACP has no free-form text input method; calling
    it under the adapter is a programmer error worth surfacing loudly rather than
    hanging on a stdin that's already owned by the JSON-RPC transport.
    """

    def __init__(self, bridge: FsBridge) -> None:
        self._bridge = bridge
        self._cleanup: Callable[[], None] = lambda: None

    def __enter__(self) -> Self:
        # pyrefly: ignore [bad-argument-type]
        self._cleanup = setup_io_context(read=self._read, confirm=self._confirm)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._cleanup()
        self._cleanup = lambda: None

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
