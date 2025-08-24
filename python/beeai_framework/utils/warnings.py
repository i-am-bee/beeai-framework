# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import sys
import types
from collections.abc import Callable
from functools import cache
from typing import Any, TypeVar

from beeai_framework.logger import Logger

logger = Logger(__name__)

T = TypeVar("T")


@cache
def _deprecated_alias(old_name: str, removal_version: str | None = None, reason: str | None = None) -> Callable[[T], T]:
    """Decorator-style function to create deprecated aliases.

    Args:
        old_name: the name of the type being deprecated.
        removal_version: version in which the deprecated type alias will be removed.
        reason: explanation for the deprecation (optional, auto-generated).

    Returns:
        A wrapper that accepts the new type being aliased as argument and returns tha type.

    Usage:
        AgentRunOutput = deprecated_alias("AgentRunOutput", removal_version="0.2")(AgentOutput)
    """

    def decorator(new_type: T) -> T:
        logger.warning(
            f"{old_name} is deprecated"
            + (f" and will be removed in version {removal_version}." if removal_version else ".")
            + (f" {reason}." if reason else f" Use {new_type.__name__} instead."),  # type: ignore[attr-defined]
            stacklevel=2,
        )
        return new_type

    return decorator


def deprecated_type_alias(
    module_name: str,
    old_name: str,
    new_type: type[T],
    /,
    removal_version: str | None = None,
    reason: str | None = None,
) -> None:
    """Dynamically create type aliases in a module.

    Args:
        module_name: the module name in which to create the deprecated alias.
        old_name: the name of the type being deprecated.
        new_type: the type to be aliased.
        removal_version: version in which the deprecated type alias will be removed.
        reason: explanation for the deprecation (optional, auto-generated).

    Usage:
        deprecated_type_alias(__name__, AgentOutput, "AgentRunOutput")
        if TYPE_CHECKING: # This will only be seen by type checkers, not at runtime
            AgentRunOutput: TypeAlias = AgentOutput
    """
    module: types.ModuleType = sys.modules[module_name]
    _orig_get_attr = module.__dict__.get("__getattr__", None)

    def _getattr(name: str) -> Any:
        if name == old_name:
            alias = _deprecated_alias(old_name, removal_version, reason)(new_type)
            module.__dict__[old_name] = alias
            return alias
        elif _orig_get_attr:
            return _orig_get_attr(name)
        else:
            raise AttributeError(f"module '{module.__name__}' has no attribute '{name}'")

    module.__dict__["__getattr__"] = _getattr
