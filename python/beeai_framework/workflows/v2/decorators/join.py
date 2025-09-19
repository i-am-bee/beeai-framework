# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from beeai_framework.workflows.v2.types import AsyncFunc


def join(func: AsyncFunc) -> AsyncFunc:
    func._is_join = True  # type: ignore[attr-defined]
    return func
