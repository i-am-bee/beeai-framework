# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from beeai_framework.context import RunContext
from beeai_framework.tools import StringToolOutput, ToolInputValidationError
from beeai_framework.tools.scratchpad import ScratchpadInput, ScratchpadTool


@pytest.fixture
def mock_context() -> RunContext:
    """Create a mock RunContext with a test run_id."""
    context = Mock(spec=RunContext)
    context.run_id = "test-run-123"
    context.conversation_id = None
    context.agent_id = None
    return context


@pytest.fixture
def tool(mock_context: RunContext) -> ScratchpadTool:
    """Create a fresh scratchpad tool instance for each test."""
    tool_instance = ScratchpadTool()
    # Initialize the session and clear any existing data
    session_id = tool_instance._get_session_id(mock_context)
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
async def test_write_key_value_with_comma_in_value(tool: ScratchpadTool) -> None:
    """Test writing a key-value pair where the value contains a comma."""
    await tool.run(
        input=ScratchpadInput(
            operation="write", content="item: milk, bread, eggs, priority: high"
        )
    )
    read_result = await tool.run(input=ScratchpadInput(operation="read"))
    assert "item: milk, bread, eggs" in read_result.result
    assert "priority: high" in read_result.result


@pytest.mark.asyncio
async def test_write_key_with_hyphens(tool: ScratchpadTool) -> None:
    """Test that keys with hyphens are correctly parsed."""
    await tool.run(
        input=ScratchpadInput(
            operation="write", content="Content-Type: application/json, user-id: 12345"
        )
    )
    read_result = await tool.run(input=ScratchpadInput(operation="read"))
    assert "Content-Type: application/json" in read_result.result
    assert "user-id: 12345" in read_result.result


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
    """Test that write operation without content raises validation error."""
    with pytest.raises(ToolInputValidationError, match="content"):
        await tool.run(input=ScratchpadInput(operation="write"))


@pytest.mark.asyncio
async def test_append_without_content_fails(tool: ScratchpadTool) -> None:
    """Test that append operation without content raises validation error."""
    # Add an entry first
    await tool.run(input=ScratchpadInput(operation="write", content="Entry"))

    # Try to append without content
    with pytest.raises(ToolInputValidationError, match="content"):
        await tool.run(input=ScratchpadInput(operation="append"))


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

    # Get the cached session ID from the tool instance
    session_id = tool._cached_session_id

    # Get scratchpad using class method
    entries = ScratchpadTool.get_scratchpad_for_session(session_id)
    assert len(entries) > 0
    assert "Test entry" in entries[0]


@pytest.mark.asyncio
async def test_clear_session_class_method(tool: ScratchpadTool) -> None:
    """Test the class method to clear a specific session."""
    # Write some data
    await tool.run(input=ScratchpadInput(operation="write", content="Test entry"))

    # Get the cached session ID from the tool instance
    session_id = tool._cached_session_id

    # Clear using async class method
    await ScratchpadTool.clear_session(session_id)

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


@pytest.mark.asyncio
async def test_session_id_requires_context() -> None:
    """Test that session ID extraction requires a valid context."""
    tool_instance = ScratchpadTool()

    # Should raise ToolInputValidationError when no context provided
    with pytest.raises(ToolInputValidationError, match="requires RunContext"):
        tool_instance._get_session_id(None)


@pytest.mark.asyncio
async def test_session_id_requires_valid_identifier() -> None:
    """Test that session ID extraction requires a valid identifier in context."""
    tool_instance = ScratchpadTool()

    # Create a context with no valid identifiers
    empty_context = Mock(spec=RunContext)
    empty_context.run_id = None
    empty_context.conversation_id = None
    empty_context.agent_id = None

    # Should raise ToolInputValidationError when no valid identifier found
    with pytest.raises(ToolInputValidationError, match="None found in context"):
        tool_instance._get_session_id(empty_context)


@pytest.mark.asyncio
async def test_session_id_caching(mock_context: RunContext) -> None:
    """Test that session ID is cached after first extraction."""
    tool_instance = ScratchpadTool()

    # First call should extract and cache
    session_id_1 = tool_instance._get_session_id(mock_context)
    assert session_id_1 == "test-run-123"
    assert tool_instance._cached_session_id == "test-run-123"

    # Modify the context
    mock_context.run_id = "different-run-456"

    # Second call should return cached value, not re-extract
    session_id_2 = tool_instance._get_session_id(mock_context)
    assert session_id_2 == "test-run-123"  # Still the original cached value


@pytest.mark.asyncio
async def test_session_id_preference_order() -> None:
    """Test that session ID extraction follows the correct preference order."""
    tool_instance = ScratchpadTool()

    # Test 1: run_id takes precedence
    context1 = Mock(spec=RunContext)
    context1.run_id = "run-123"
    context1.conversation_id = "conv-456"
    context1.agent_id = "agent-789"
    assert tool_instance._get_session_id(context1) == "run-123"

    # Test 2: conversation_id is second
    tool_instance2 = ScratchpadTool()
    context2 = Mock(spec=RunContext)
    context2.run_id = None
    context2.conversation_id = "conv-456"
    context2.agent_id = "agent-789"
    assert tool_instance2._get_session_id(context2) == "conv-456"

    # Test 3: agent_id is last
    tool_instance3 = ScratchpadTool()
    context3 = Mock(spec=RunContext)
    context3.run_id = None
    context3.conversation_id = None
    context3.agent_id = "agent-789"
    assert tool_instance3._get_session_id(context3) == "agent-789"


@pytest.mark.asyncio
async def test_concurrent_writes_do_not_corrupt_state(tool: ScratchpadTool) -> None:
    """Test that concurrent writes do not corrupt the scratchpad."""
    import asyncio

    async def write_entry(content: str) -> None:
        await tool.run(input=ScratchpadInput(operation="write", content=content))

    # With key-value merging, each write will update the value for 'entry'.
    # We run them concurrently to test for race conditions.
    tasks = [write_entry(f"entry: {i}") for i in range(10)]
    await asyncio.gather(*tasks)

    result = await tool.run(input=ScratchpadInput(operation="read"))

    # The final state should contain one of the written values due to merging.
    # Without proper locking, the final state could be unpredictable.
    assert "entry:" in result.result
    # Check that it's a single consolidated entry
    assert result.result.count("entry:") == 1
    # Verify the value is one of the expected outcomes from concurrent writes
    # The exact value depends on the order of completion, but it should be one of the 'entry: i' values
    found_value = result.result.split("entry: ")[1].split()[0]
    assert found_value.isdigit() and 0 <= int(found_value) < 10
