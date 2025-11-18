# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Any

from beeai_framework.adapters.openai.backend.chat import OpenAIChatModel
from beeai_framework.adapters.openai.backend.embedding import OpenAIEmbeddingModel

__all__ = ["OpenAIChatModel", "OpenAIEmbeddingModel"]


def __getattr__(name: str) -> Any:
    if name in {"OpenAIServer", "OpenAIServerConfig", "OpenAIServerMetadata"}:
        import beeai_framework.adapters.openai.serve.server as serve

        warnings.warn(
            f"Please import {name} from beeai_framework.adapters.openai.serve.server instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return getattr(serve, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
