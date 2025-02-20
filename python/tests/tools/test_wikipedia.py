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
    WikipediaSearchTool,
    WikipediaSearchToolInput,
    WikipediaSearchToolOutput,
)


@pytest.fixture
def tool() -> WikipediaSearchTool:
    return WikipediaSearchTool()


@pytest.mark.unit
def test_call_invalid_input_type(tool: WikipediaSearchTool) -> None:
    with pytest.raises(ToolInputValidationError):
        tool.run(input={"search": "Bee"})


@pytest.mark.e2e
def test_output(tool: WikipediaSearchTool) -> None:
    result = tool.run(input=WikipediaSearchToolInput(query="bee"))
    assert type(result) is WikipediaSearchToolOutput
    assert "Bees are winged insects closely related to wasps and ants" in result.get_text_content()
