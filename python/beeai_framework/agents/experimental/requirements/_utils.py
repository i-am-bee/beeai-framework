# Copyright 2025 © BeeAI a Series of LF Projects, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, TypeVar

from beeai_framework.errors import FrameworkError
from beeai_framework.tools import AnyTool, Tool

T = TypeVar("T", bound=Any)

TargetType = str | type[AnyTool] | AnyTool
MultiTargetType = list[TargetType] | TargetType


def _extract_targets(
    target: MultiTargetType | None,
) -> set[str | type | AnyTool]:
    targets = target if isinstance(target, list) else [target] if target is not None else []
    return set(targets)


_HaystackValue = str | type | AnyTool


def _assert_all_rules_found(
    targets: set[_HaystackValue],
    tools: list[AnyTool],
) -> None:
    for target in targets:
        for tool in tools:
            if _target_seen_in(tool, {target}):
                break
        else:
            raise ValueError(
                f"Tool '{target}' is specified as 'source', 'before', 'after' or 'force_after' but not found."
            )


def _target_seen_in(
    target: AnyTool | None,
    haystack: set[_HaystackValue] | _HaystackValue,
) -> _HaystackValue | None:
    if target is None:
        return None

    if not isinstance(haystack, set):
        haystack = {haystack}

    for needle in haystack:
        if isinstance(needle, str) and needle == target.name:
            return needle
        if isinstance(needle, type) and isinstance(target, needle):
            return needle
        if isinstance(needle, Tool) and needle is target:
            return needle

    return None


def _assert_targets_exist(
    targets: list[AnyTool],
    allowed: set[str | type | AnyTool],
) -> None:
    """Checks if all in received are in allowed."""

    if not allowed:
        return

    for r in targets:
        for t in allowed:
            if _target_seen_in(r, {t}):
                break
        else:
            raise FrameworkError(f"Tool '{r}' was not found in {allowed}!", is_fatal=True, is_retryable=False)


def _extract_target_name(target: TargetType) -> str:
    if isinstance(target, str):
        return target
    elif isinstance(target, Tool):
        return target.name
    else:
        return target.__qualname__
