# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import re
from typing import TYPE_CHECKING, Optional

from beeai_framework.emitter.errors import EmitterError
from beeai_framework.utils.lists import cast_list

if TYPE_CHECKING:
    from beeai_framework.context import RunInstance
    from beeai_framework.emitter import Emitter, EventMeta, Matcher, MatcherFn


def assert_valid_name(name: str) -> None:
    if not name or not re.match("^[a-zA-Z0-9_]+$", name):
        raise EmitterError(
            f"Event name or a namespace part must contain only letters, numbers or underscores: {name}",
        )


def assert_valid_namespace(path: list[str]) -> None:
    for part in path:
        assert_valid_name(part)


def create_internal_event_matcher(
    name: str | list[str],
    instance: Optional["RunInstance"] = None,
    *,
    parent_run_id: str | None = None,
    excluded_instance: object | None = None,
    excluded_emitter: Optional["Emitter"] = None,
) -> "MatcherFn":
    # pyrefly: ignore [bad-assignment]
    allowed_names: list[str] = cast_list(name)

    def matcher(event: "EventMeta") -> bool:
        if parent_run_id is not None and (not event.trace or event.trace.parent_run_id != parent_run_id):
            return False

        if not bool(event.context.get("internal", False)):
            return False

        if event.name not in allowed_names:
            return False

        if event.creator.instance is excluded_instance:  # type: ignore
            return False

        if event.creator.emitter is excluded_emitter:  # type: ignore
            return False

        if instance is not None:
            return (
                event.path == ".".join(["run", *instance.emitter.namespace, event.name])
                and event.creator.instance is instance  # type: ignore
            )

        return True

    return matcher


def create_event_matcher(name: str, instance: "RunInstance", *, parent_run_id: str | None = None) -> "Matcher":
    def matcher(event: "EventMeta") -> bool:
        if parent_run_id is not None and (not event.trace or event.trace.parent_run_id != parent_run_id):
            return False

        return event.name == name and event.creator is instance

    return matcher
