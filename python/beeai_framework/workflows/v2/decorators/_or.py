# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from beeai_framework.workflows.v2.types import AsyncMethod, AsyncMethodSet


def _or(*methods: AsyncMethod | str) -> AsyncMethodSet:
    asm = AsyncMethodSet()

    for method in methods:
        if isinstance(method, str):
            asm.methods.append(method)
        elif callable(method):
            asm.methods.append(method.__name__)

    asm.condition = "or"
    return asm
