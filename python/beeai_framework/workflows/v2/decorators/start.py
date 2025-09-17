# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Any, cast

from beeai_framework.workflows.v2.types import AsyncFunc


def start(func: AsyncFunc) -> AsyncFunc:
    class Wrapper:
        def __init__(self, fn: AsyncFunc) -> None:
            self._func = fn
            self._is_start = True  # instance marker

        async def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return await self._func(*args, **kwargs)

    wrapper_instance = Wrapper(func)
    functools.update_wrapper(wrapper_instance, func)
    return cast(AsyncFunc, wrapper_instance)
