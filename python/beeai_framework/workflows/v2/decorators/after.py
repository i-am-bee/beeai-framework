# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

from beeai_framework.workflows.v2.types import AsyncMethod, AsyncMethodSet


def after(dependency: str | AsyncMethod | AsyncMethodSet) -> Callable[[AsyncMethod], AsyncMethod]:
    def decorator(func: AsyncMethod) -> AsyncMethod:
        func._is_step = True  # type: ignore[attr-defined]
        func._dependency = dependency  # type: ignore[attr-defined]
        return func

    return decorator
