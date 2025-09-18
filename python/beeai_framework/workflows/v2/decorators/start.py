# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0


from beeai_framework.workflows.v2.workflow_types import AsyncFunc


def start(func: AsyncFunc) -> AsyncFunc:
    func._is_step = True  # type: ignore[attr-defined]
    func._is_start = True  # type: ignore[attr-defined]
    return func
