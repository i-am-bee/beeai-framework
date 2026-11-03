# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

"""Helpers that let the MCP integration run on both the MCP Python SDK 1.x and 2.x lines."""

from typing import Any

from mcp.types import CallToolResult
from pydantic.alias_generators import to_camel

__all__ = ["MCP_V2", "get_field"]

# mcp 2.x renamed the protocol model fields from camelCase to snake_case (the wire format is unchanged).
MCP_V2: bool = "is_error" in CallToolResult.model_fields


def get_field(model: Any, name: str) -> Any:
    """Read a protocol model field by its snake_case (2.x) name, using the camelCase (1.x) name on 1.x."""
    return getattr(model, name if MCP_V2 else to_camel(name))
