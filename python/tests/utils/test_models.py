import pytest
from pydantic import BaseModel
from beeai_framework.utils.models import to_any_model

class Dog(BaseModel):
    name: str

class Cat(BaseModel):
    name: str

def test_to_any_model_raises_on_invalid():
    with pytest.raises(ValueError):
        to_any_model([Dog, Cat], {"age": 5})