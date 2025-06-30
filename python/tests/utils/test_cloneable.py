# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0


import pytest
from pydantic import BaseModel

from beeai_framework.utils.cloneable import Cloneable

"""
Utility functions and classes
"""


class DefaultUser:
    def __init__(self, name: str = "", age: int = 0, email: str | None = None) -> None:
        self.name = name
        self.age = age
        self.email = email

    async def clone(self) -> "DefaultUser":
        return DefaultUser(self.name, self.age, self.email)


class BaseModelUser(BaseModel):
    name: str
    age: int
    email: str | None = None

    async def clone(self) -> "BaseModelUser":
        return BaseModelUser(**self.model_dump())


@pytest.fixture
def default_user() -> DefaultUser:
    return DefaultUser("John Doe", 42)


@pytest.fixture
def base_model_user() -> BaseModelUser:
    return BaseModelUser(name="John Doe", age=42)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_clone(default_user: DefaultUser) -> None:
    assert isinstance(default_user, Cloneable)

    cloned_user = await default_user.clone()

    assert isinstance(cloned_user, Cloneable)
    assert cloned_user.name == default_user.name
    assert cloned_user.age == default_user.age
    assert cloned_user.email == default_user.email

    cloned_user.name = "Jane Doe"
    cloned_user.email = "jane.doe@domain.com"

    assert cloned_user.name != default_user.name
    assert cloned_user.email != default_user.email


@pytest.mark.unit
@pytest.mark.asyncio
async def test_clone_basemodel(base_model_user: BaseModelUser) -> None:
    assert isinstance(base_model_user, Cloneable)

    cloned_user = await base_model_user.clone()

    assert isinstance(cloned_user, Cloneable)
    assert cloned_user.name == base_model_user.name
    assert cloned_user.age == base_model_user.age
    assert cloned_user.email == base_model_user.email

    cloned_user.name = "Jane Doe"
    cloned_user.email = "jane.doe@domain.com"

    assert cloned_user.name != base_model_user.name
    assert cloned_user.email != base_model_user.email
