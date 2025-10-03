# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable

from beeai_framework.workflows.v2.types import AsyncFunc


def retry(n: int = 1) -> Callable[[AsyncFunc], AsyncFunc]:
    def decorator(func: AsyncFunc) -> AsyncFunc:
        func._retries = n  # type: ignore[attr-defined]
        return func

    return decorator
