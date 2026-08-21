# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from beeai_framework.tools.search.feedo import FeedoSearchTool, FeedoToolInput


@pytest.fixture
def feedo_memory_mock():
    with patch("beeai_framework.tools.search.feedo.tool.FeedoMemory") as mock_memory_cls:
        mock_instance = MagicMock()
        mock_memory_cls.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def feedo_tool(feedo_memory_mock):
    # Mock OS environment to bypass missing key validation
    with patch.dict("os.environ", {"FEEDO_USAGE_KEY": "dummy_key"}):
        tool = FeedoSearchTool()
        return tool


@pytest.mark.asyncio
async def test_feedo_add(feedo_tool, feedo_memory_mock):
    feedo_memory_mock.add.return_value = "mem_123"
    
    input_data = FeedoToolInput(action="add", query="Hello World", topic="greeting")
    output = await feedo_tool.run(input_data)
    
    feedo_memory_mock.add.assert_called_once_with("Hello World", metadata={"topic": "greeting"})
    assert "mem_123" in output.result


@pytest.mark.asyncio
async def test_feedo_search(feedo_tool, feedo_memory_mock):
    feedo_memory_mock.search.return_value = [{"text": "Hello World"}]
    
    input_data = FeedoToolInput(action="search", query="Hello")
    output = await feedo_tool.run(input_data)
    
    feedo_memory_mock.search.assert_called_once_with("Hello", limit=5)
    assert output.memories == [{"text": "Hello World"}]
    assert "Found 1" in output.result


@pytest.mark.asyncio
async def test_feedo_search_empty(feedo_tool, feedo_memory_mock):
    feedo_memory_mock.search.return_value = []
    
    input_data = FeedoToolInput(action="search", query="Unknown")
    output = await feedo_tool.run(input_data)
    
    feedo_memory_mock.search.assert_called_once_with("Unknown", limit=5)
    assert output.memories == []
    assert "No relevant memories found" in output.result


@pytest.mark.asyncio
async def test_feedo_update(feedo_tool, feedo_memory_mock):
    feedo_memory_mock.update.return_value = "mem_456"
    
    input_data = FeedoToolInput(action="update", query="New text", memory_id="mem_123")
    output = await feedo_tool.run(input_data)
    
    feedo_memory_mock.update.assert_called_once_with("mem_123", "New text")
    assert "mem_456" in output.result


@pytest.mark.asyncio
async def test_feedo_delete(feedo_tool, feedo_memory_mock):
    feedo_memory_mock.delete.return_value = None
    
    input_data = FeedoToolInput(action="delete", memory_id="mem_123")
    output = await feedo_tool.run(input_data)
    
    feedo_memory_mock.delete.assert_called_once_with("mem_123")
    assert "mem_123" in output.result
