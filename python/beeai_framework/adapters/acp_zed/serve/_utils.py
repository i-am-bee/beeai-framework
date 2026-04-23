# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

try:
    from acp.schema import (
        AudioContentBlock,
        EmbeddedResourceContentBlock,
        ImageContentBlock,
        ResourceContentBlock,
        TextContentBlock,
    )
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [acp_zed] not found.\nRun 'pip install \"beeai-framework[acp-zed]\"' to install."
    ) from e

from beeai_framework.backend import UserMessage
from beeai_framework.backend.message import AnyMessage

PromptBlock = (
    TextContentBlock | ImageContentBlock | AudioContentBlock | ResourceContentBlock | EmbeddedResourceContentBlock
)


def _block_text(block: Any) -> str:
    if isinstance(block, dict):
        if block.get("type") == "text":
            return str(block.get("text", ""))
        resource = block.get("resource") if block.get("type") == "resource" else None
        if isinstance(resource, dict):
            return str(resource.get("text") or "")
        return ""
    text = getattr(block, "text", None)
    if isinstance(text, str):
        return text
    inner = getattr(getattr(block, "resource", None), "text", None)
    return inner if isinstance(inner, str) else ""


def acp_zed_prompt_to_framework_msgs(prompt: list[PromptBlock]) -> list[AnyMessage]:
    """Collapse ACP prompt content blocks into a single framework `UserMessage`.

    Text and embedded-resource text blocks are joined. Image/audio blocks are dropped —
    pass-through will land when BeeAI's `UserMessage` multi-part support stabilizes
    across backends.
    """
    parts = [text for block in prompt if (text := _block_text(block))]
    return [UserMessage("\n".join(parts) if parts else "")]
