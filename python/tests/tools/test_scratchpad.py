# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.tools import StringToolOutput, ToolInputValidationError
from beeai_framework.tools.scratchpad import ScratchpadInput, ScratchpadTool


@pytest.fixture
def tool() -> ScratchpadTool:
    """Create a fresh scratchpad tool instance for each test."""
    tool_instance = ScratchpadTool()
    # Clear any existing data from previous tests
    session_id = tool_instance._get_session_id()
    if session_id in ScratchpadTool._scratchpads:
        ScratchpadTool._scratchpads[session_id] = []
    return tool_instance


"""
Unit Tests
"""


@pytest.mark.asyncio
async def test_read_empty_scratchpad(tool: ScratchpadTool) -> None:
    """Test reading an empty scratchpad."""
    result = await tool.run(input=ScratchpadInput(operation="read"))
    assert isinstance(result, StringToolOutput)
    assert "empty" in result.result.lower()


@pytest.mark.asyncio
async def test_write_to_scratchpad(tool: ScratchpadTool) -> None:
    """Test writing an entry to the scratchpad."""
    result = await tool.run(
        input=ScratchpadInput(operation="write", content="Test action completed")
    )
    assert isinstance(result, StringToolOutput)
    assert "added" in result.result.lower() or "updated" in result.result.lower()


@pytest.mark.asyncio
async def test_write_and_read(tool: ScratchpadTool) -> None:
    """Test writing and then reading from the scratchpad."""
    # Write an entry
    await tool.run(input=ScratchpadInput(operation="write", content="First entry"))

    # Read it back
    result = await tool.run(input=ScratchpadInput(operation="read"))
    assert isinstance(result, StringToolOutput)
    assert "First entry" in result.result


@pytest.mark.asyncio
async def test_write_key_value_pairs(tool: ScratchpadTool) -> None:
    """Test writing key-value pairs to the scratchpad."""
    result = await tool.run(
        input=ScratchpadInput(
            operation="write", content="city: New York, date: 2025-01-27"
        )
    )
    assert isinstance(result, StringToolOutput)

    # Read it back
    read_result = await tool.run(input=ScratchpadInput(operation="read"))
    assert "city: New York" in read_result.result
    assert "date: 2025-01-27" in read_result.result


@pytest.mark.asyncio
async def test_write_merge_key_value_pairs(tool: ScratchpadTool) -> None:
    """Test that writing new key-value pairs merges with existing ones."""
    # Write first set of pairs
    await tool.run(input=ScratchpadInput(operation="write", content="city: Boston"))

    # Write second set of pairs (should merge)
    await tool.run(input=ScratchpadInput(operation="write", content="date: 2025-01-28"))

    # Read it back
    result = await tool.run(input=ScratchpadInput(operation="read"))
    assert "city: Boston" in result.result
    assert "date: 2025-01-28" in result.result


@pytest.mark.asyncio
async def test_append_to_scratchpad(tool: ScratchpadTool) -> None:
    """Test appending to the last entry."""
    # Write initial entry
    await tool.run(input=ScratchpadInput(operation="write", content="Initial entry"))

    # Append to it
    result = await tool.run(
        input=ScratchpadInput(operation="append", content="- additional info")
    )
    assert isinstance(result, StringToolOutput)
    assert "appended" in result.result.lower()

    # Read it back
    read_result = await tool.run(input=ScratchpadInput(operation="read"))
    assert "Initial entry - additional info" in read_result.result


@pytest.mark.asyncio
async def test_append_without_entry_fails(tool: ScratchpadTool) -> None:
    """Test that appending without an existing entry returns an error."""
    result = await tool.run(
        input=ScratchpadInput(operation="append", content="some content")
    )
    assert isinstance(result, StringToolOutput)
    assert "no entry" in result.result.lower()


@pytest.mark.asyncio
async def test_clear_scratchpad(tool: ScratchpadTool) -> None:
    """Test clearing the scratchpad."""
    # Write some entries
    await tool.run(input=ScratchpadInput(operation="write", content="Entry 1"))
    await tool.run(input=ScratchpadInput(operation="write", content="Entry 2"))

    # Clear the scratchpad
    result = await tool.run(input=ScratchpadInput(operation="clear"))
    assert isinstance(result, StringToolOutput)
    assert "cleared" in result.result.lower()

    # Verify it's empty
    read_result = await tool.run(input=ScratchpadInput(operation="read"))
    assert "empty" in read_result.result.lower()


@pytest.mark.asyncio
async def test_write_without_content_fails(tool: ScratchpadTool) -> None:
    """Test that write operation without content returns an error."""
    result = await tool.run(input=ScratchpadInput(operation="write"))
    assert isinstance(result, StringToolOutput)
    assert "error" in result.result.lower()
    assert "content" in result.result.lower()


@pytest.mark.asyncio
async def test_append_without_content_fails(tool: ScratchpadTool) -> None:
    """Test that append operation without content returns an error."""
    # Add an entry first
    await tool.run(input=ScratchpadInput(operation="write", content="Entry"))

    # Try to append without content
    result = await tool.run(input=ScratchpadInput(operation="append"))
    assert isinstance(result, StringToolOutput)
    assert "error" in result.result.lower()
    assert "content" in result.result.lower()


@pytest.mark.asyncio
async def test_invalid_operation(tool: ScratchpadTool) -> None:
    """Test that an invalid operation is handled properly."""
    with pytest.raises(ToolInputValidationError):
        await tool.run(input={"operation": "invalid_op"})


@pytest.mark.asyncio
async def test_get_scratchpad_for_session(tool: ScratchpadTool) -> None:
    """Test the class method to get scratchpad for a session."""
    # Write some data
    await tool.run(input=ScratchpadInput(operation="write", content="Test entry"))

    # Get the session ID
    session_id = tool._get_session_id()

    # Get scratchpad using class method
    entries = ScratchpadTool.get_scratchpad_for_session(session_id)
    assert len(entries) > 0
    assert "Test entry" in entries[0]


@pytest.mark.asyncio
async def test_clear_session_class_method(tool: ScratchpadTool) -> None:
    """Test the class method to clear a specific session."""
    # Write some data
    await tool.run(input=ScratchpadInput(operation="write", content="Test entry"))

    # Get the session ID
    session_id = tool._get_session_id()

    # Clear using class method
    ScratchpadTool.clear_session(session_id)

    # Verify it's empty
    entries = ScratchpadTool.get_scratchpad_for_session(session_id)
    assert len(entries) == 0


@pytest.mark.asyncio
async def test_call_with_dict_input(tool: ScratchpadTool) -> None:
    """Test calling the tool with dictionary input instead of model."""
    result = await tool.run(input={"operation": "write", "content": "Dict entry"})
    assert isinstance(result, StringToolOutput)

    read_result = await tool.run(input={"operation": "read"})
    assert "Dict entry" in read_result.result


@pytest.mark.asyncio
async def test_missing_operation_field(tool: ScratchpadTool) -> None:
    """Test that missing operation field raises validation error."""
    with pytest.raises(ToolInputValidationError):
        await tool.run(input={})
