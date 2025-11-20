# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from beeai_framework.workflows.v2.types import AsyncMethod


def fork(func: AsyncMethod) -> AsyncMethod:
    func._is_fork = True  # type: ignore[attr-defined]
    return func
