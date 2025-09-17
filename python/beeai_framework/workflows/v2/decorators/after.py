# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import functools
from collections.abc import Callable
from typing import Any, cast

from beeai_framework.workflows.v2.types import AsyncFunc


def after(*dependencies: AsyncFunc) -> Callable[[AsyncFunc], AsyncFunc]:
    def decorator(func: AsyncFunc) -> AsyncFunc:
        class Wrapper:
            def __init__(self, fn: AsyncFunc, deps: tuple[AsyncFunc, ...]) -> None:
                self._func = fn
                self._dependencies = list(deps)  # instance variable

            async def __call__(self, *args: Any, **kwargs: Any) -> Any:
                return await self._func(*args, **kwargs)

            @property
            def dependencies(self) -> list[AsyncFunc]:
                return self._dependencies

        wrapper_instance = Wrapper(func, dependencies)
        functools.update_wrapper(wrapper_instance, func)
        return cast(AsyncFunc, wrapper_instance)

    return decorator
