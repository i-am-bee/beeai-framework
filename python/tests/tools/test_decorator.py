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

from beeai_framework.tools import StringToolOutput, tool

"""
Unit Tests
"""


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_annotation() -> None:
    @tool
    def test_tool(query: str) -> str:
        """
        Search factual and historical information, including biography, history, politics, geography, society, culture,
        science, technology, people, animal species, mathematics, and other subjects.

        Args:
            query: The topic or question to search for on Wikipedia.

        Returns:
            The information found via searching Wikipedia.
        """
        return query

    query = "Hello!"
    result: StringToolOutput = await test_tool.run({"query": query})
    assert result.get_text_content() == query


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_annotation_no_params() -> None:
    @tool
    def test_tool() -> str:
        """
        Search factual and historical information, including biography, history, politics, geography, society, culture,
        science, technology, people, animal species, mathematics, and other subjects.
        """
        return "Hello!"

    result: StringToolOutput = await test_tool.run({})
    assert result.get_text_content() == "Hello!"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_annotation_empty_desc() -> None:
    @tool
    def test_tool() -> str:
        """"""
        return "Hello!"

    result: StringToolOutput = await test_tool.run({})
    assert result.get_text_content() == "Hello!"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_annotation_no_desc() -> None:
    with pytest.raises(ValueError):  # No description provided

        @tool
        def test_tool(query: str) -> str:
            return query

        await test_tool.run({"query": "Hello!"})
