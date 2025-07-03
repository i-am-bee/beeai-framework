# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
from pydantic import BaseModel

from beeai_framework.emitter import Emitter, EventMeta
from beeai_framework.tools import StringToolOutput, tool

"""
Unit Tests
"""


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_emitter() -> None:
    async def process_agent_events(event_data: Any, event_meta: EventMeta) -> None:
        print(
            event_meta.name,
            event_meta.path,
            event_data.model_dump() if isinstance(event_data, BaseModel) else event_data,
        )

    # Observe the agent
    async def observer(emitter: Emitter) -> None:
        emitter.on("*.*", process_agent_events)

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
    result: StringToolOutput = await test_tool.run({"query": query}).observe(observer)
    assert result.get_text_content() == query
