# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
from typing import Protocol, TypeVar, runtime_checkable

T = TypeVar("T", bound="Cloneable")


@runtime_checkable
class Cloneable(Protocol):
    async def clone(self: T) -> T: ...


TC = TypeVar("TC", bound=type)


def clone_class(cls: TC) -> TC:
    return type(f"{cls.__name__}Copy", cls.__bases__, dict(cls.__dict__))  # type: ignore
