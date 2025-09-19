# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

from beeai_framework.workflows.v2.types import AsyncFunc, DependencyType

Dependency = str | AsyncFunc


def after(*dependencies: Dependency, type: DependencyType = "AND") -> Callable[[AsyncFunc], AsyncFunc]:
    def decorator(func: AsyncFunc) -> AsyncFunc:
        func._is_step = True  # type: ignore[attr-defined]
        func._dependencies = list(dependencies)  # type: ignore[attr-defined]
        func._dependency_type = type  # type: ignore[attr-defined]
        return func

    return decorator
