# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
