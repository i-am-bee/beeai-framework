import pytest
from pydantic import BaseModel
from beeai_framework.utils.models import to_any_model

<<<<<<< Updated upstream
class Dog(BaseModel):
    name: str

class Cat(BaseModel):
    name: str

def test_to_any_model_raises_on_invalid():
    with pytest.raises(ValueError):
        to_any_model([Dog, Cat], {"age": 5})
=======

class Dog(BaseModel):
    name: str


class Cat(BaseModel):
    name: str


def test_to_any_model_raises_on_invalid():
    with pytest.raises(ValueError, match="Failed to create a model instance from the passed object!\nDog\nCat"):
        to_any_model([Dog, Cat], {"age": 5})
        
>>>>>>> Stashed changes
