# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Any


def __getattr__(name: str) -> Any:
    if name in {"A2AServer", "A2AServerConfig"}:
        import beeai_framework.adapters.a2a.serve.server as serve

        warnings.warn(
            f"Please import {name} from beeai_framework.adapters.a2a.serve.server instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return getattr(serve, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
