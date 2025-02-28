# Copyright 2025 IBM Corp.
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


import pytest

from beeai_framework.tools import ToolInputValidationError
from beeai_framework.tools.search.wikipedia import (
    WikipediaTool,
    WikipediaToolInput,
    WikipediaToolOutput,
)

"""
Utility functions and classes
"""


@pytest.fixture
def tool() -> WikipediaTool:
    return WikipediaTool()


"""
E2E Tests
"""


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_call_invalid_input_type(tool: WikipediaTool) -> None:
    with pytest.raises(ToolInputValidationError):
        await tool.run(input={"search": "Bee"})


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_output(tool: WikipediaTool) -> None:
    result = await tool.run(input=WikipediaToolInput(query="bee"))
    assert type(result) is WikipediaToolOutput
    assert "Bees are winged insects closely related to wasps and ants" in result.get_text_content()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_full_text_output(tool: WikipediaTool) -> None:
    result = await tool.run(input=WikipediaToolInput(query="bee", full_text=True))
    assert type(result) is WikipediaToolOutput
    assert "n-triscosane" in result.get_text_content()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_section_titles(tool: WikipediaTool) -> None:
    result = await tool.run(input=WikipediaToolInput(query="bee", section_titles=True))
    assert type(result) is WikipediaToolOutput
    assert "Characteristics" in result.get_text_content()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_alternate_language(tool: WikipediaTool) -> None:
    result = await tool.run(input=WikipediaToolInput(query="bee", language="fr"))
    assert type(result) is WikipediaToolOutput
    assert "Les abeilles (Anthophila) forment un clade d'insectes" in result.get_text_content()
