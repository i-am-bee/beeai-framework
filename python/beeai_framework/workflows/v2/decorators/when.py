# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

from beeai_framework.workflows.v2.types import AsyncFunc

Dependency = str | AsyncFunc

Predicate = Callable[..., bool]


def when(predicate: Predicate) -> Callable[[AsyncFunc], AsyncFunc]:
    """
    Async decorator: runs the async function only if `predicate` returns True.
    """

    def decorator(func: AsyncFunc) -> AsyncFunc:
        func._when_predicate = predicate  # type: ignore[attr-defined]
        return func

    return decorator
