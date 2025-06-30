# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Optional

from pydantic import ValidationError

from beeai_framework.errors import FrameworkError

if TYPE_CHECKING:
    from beeai_framework.tools.tool import AnyTool


class ToolError(FrameworkError):
    def __init__(
        self,
        message: str = "Tool Error",
        *,
        cause: Exception | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, cause=cause, context=context)

    @classmethod
    def ensure(
        cls,
        error: Exception,
        *,
        message: str | None = None,
        context: dict[str, Any] | None = None,
        tool: Optional["AnyTool"] = None,
    ) -> "FrameworkError":
        tool_context = {"name": tool.name} if tool is not None else {}
        tool_context.update(context) if context is not None else None
        return super().ensure(error, message=message, context=tool_context)


class ToolInputValidationError(ToolError):
    def __init__(
        self,
        message: str = "Tool Input Validation Error",
        *,
        cause: ValidationError | ValueError | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, cause=cause, context=context)
