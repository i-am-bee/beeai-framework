# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from beeai_framework.agents.experimental.rag.agent import RAGAgent, RagAgentRunInput


# lazy import deprecated aliases to prevent unecessary warnings
def __getattr__(name: str) -> Any:
    if name == "RAGAgentRunOutput":
        from beeai_framework.agents.experimental.rag.agent import RAGAgentRunOutput

        return RAGAgentRunOutput
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["RAGAgent", "RAGAgentRunOutput", "RagAgentRunInput"]
