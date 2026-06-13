# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import BaseModel
from beeai_framework.utils.models import to_any_model


class Dog(BaseModel):
    name: str


class Cat(BaseModel):
    name: str


def test_to_any_model_raises_on_invalid():
    with pytest.raises(ValueError, match="Failed to create a model instance from the passed object!\nDog\nCat"):
        to_any_model([Dog, Cat], {"age": 5})