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

from enum import StrEnum

import pytest

from beeai_framework.emitter import EventMeta
from beeai_framework.parsers.field import ParserField
from beeai_framework.parsers.line_prefix import (
    LinePrefixParser,
    LinePrefixParserError,
    LinePrefixParserNode,
    LinePrefixParserOptions,
    LinePrefixParserUpdate,
)
from beeai_framework.utils.strings import split_string

"""
Utility functions and classes
"""


def get_fallback_parser() -> LinePrefixParser:
    config = {
        "thought": LinePrefixParserNode(
            prefix="Thought:",
            field=ParserField.from_type(str),
            is_start=True,
            next=["final_answer"],
        ),
        "final_answer": LinePrefixParserNode(
            prefix="Final Answer:",
            field=ParserField.from_type(str),
            is_end=True,
            next=[],
        ),
    }

    def fallback(value: str) -> list[dict[str, str]]:
        return [
            {"key": "thought", "value": "I now know the final answer."},
            {"key": "final_answer", "value": value},
        ]

    return LinePrefixParser(config, options=LinePrefixParserOptions(fallback=fallback))


"""
Unit Tests
"""


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("chunk_size", [1, 5, 20, 50, 100, 1000])
async def test_handles_arbitrary_chunk_size(chunk_size: int) -> None:
    class ToolNames(StrEnum):
        Google = "Google"

    config = {
        "thought": LinePrefixParserNode(
            prefix="Thought:", next=["tool_name", "final_answer"], is_start=True, field=ParserField.from_type(str)
        ),
        "tool_name": LinePrefixParserNode(
            prefix="Tool Name:",
            next=["tool_caption"],
            field=ParserField.from_type(ToolNames),
        ),
        "tool_caption": LinePrefixParserNode(
            prefix="Tool Caption:", next=["tool_input"], field=ParserField.from_type(str)
        ),
        "tool_input": LinePrefixParserNode(
            prefix="Tool Input:", next=["tool_output"], field=ParserField.from_type(dict, default={})
        ),
        "tool_output": LinePrefixParserNode(
            prefix="Tool Output:",
            next=["final_answer"],
            is_end=True,
            field=ParserField.from_type(dict, default={}),
        ),
        "final_answer": LinePrefixParserNode(
            prefix="Final Answer:", next=[], is_start=True, is_end=True, field=ParserField.from_type(str)
        ),
    }
    parser = LinePrefixParser(config)

    # Capture update events.
    updates: dict[str, dict[str, str]] = {"partial": {}, "final": {}}

    def on_update(data: LinePrefixParserUpdate, event: EventMeta) -> None:
        updates["final"][data.key] = data.field.raw

    def on_partial_update(data: LinePrefixParserUpdate, event: EventMeta) -> None:
        updates["partial"][data.key] = data.field.raw

    parser.emitter.on("update", on_update)
    parser.emitter.on("partial_update", on_partial_update)

    text = (
        "Thought: I don't know who is president of USA.\n"
        "But I can use GoogleSearch to find out.\n"
        "Tool Name: Google\n"
        "Tool Caption: Searching for USA president\n"
        'Tool Input: {"query":"USA president"}\n'
        'Tool Output: {"answer":"Joe Biden"}\n'
        "Final Answer: Joe Biden is the current president of USA."
    )
    chunks = split_string(text, size=chunk_size, overlap=0)
    for chunk in chunks:
        await parser.add(chunk)
    state = await parser.end()

    expected_state = {
        "thought": "I don't know who is president of USA.\nBut I can use GoogleSearch to find out.",
        "tool_name": "Google",
        "tool_caption": "Searching for USA president",
        "tool_input": {"query": "USA president"},
        "tool_output": {"answer": "Joe Biden"},
        "final_answer": "Joe Biden is the current president of USA.",
    }
    assert state == expected_state
    assert updates["partial"] == updates["final"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_process() -> None:
    parser = get_fallback_parser()
    await parser.add("2+2=4")
    state = await parser.end()
    expected = {
        "thought": "I now know the final answer.",
        "final_answer": "2+2=4",
    }
    assert state == expected


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_process_2() -> None:
    parser = get_fallback_parser()
    await parser.add("A\nB\nC")
    state = await parser.end()
    expected = {
        "thought": "I now know the final answer.",
        "final_answer": "A\nB\nC",
    }
    assert state == expected


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_process_3() -> None:
    parser = get_fallback_parser()
    await parser.add("Thought")
    state = await parser.end()
    expected = {
        "thought": "I now know the final answer.",
        "final_answer": "Thought",
    }
    assert state == expected


@pytest.mark.asyncio
@pytest.mark.unit
async def test_prevents_processing_potential_prefix() -> None:
    config = {
        "final_answer": LinePrefixParserNode(
            prefix="Final Answer:",
            field=ParserField.from_type(str),
            is_start=True,
            is_end=True,
            next=["final_detail"],
        ),
        "final_detail": LinePrefixParserNode(
            prefix="Final Detail:",
            field=ParserField.from_type(str),
            is_start=True,
            is_end=True,
            next=["final_answer"],
        ),
    }
    parser = LinePrefixParser(config)
    deltas = []

    async def on_partial_update(data: LinePrefixParserUpdate, event: EventMeta) -> None:
        # Only updates for the "final_answer" node are expected.
        assert data.key == "final_answer"
        deltas.append(data.delta)

    parser.emitter.on("partial_update", on_partial_update)
    await parser.add("Final Answer: 4\n")
    await parser.add("\n")
    assert "".join(deltas) == "4\n"
    assert len(deltas) == 2

    await parser.add("Final ")
    assert len(deltas) == 2
    await parser.add("Detail")
    assert len(deltas) == 2
    await parser.end()
    assert "".join(deltas) == "4\n\nFinal Detail"
    assert len(deltas) == 3


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("should_throw", [True, False])
async def test_interprets_new_lines(should_throw: bool) -> None:
    input_str = "Thought: Summarize the discussion. Final Answer: The discussion thread is about ..."
    config = {
        "thought": LinePrefixParserNode(
            prefix="Thought:",
            next=["final_answer"],
            is_start=True,
            is_end=not should_throw,  # invert is_end based on the parameter
            field=ParserField.from_type(str),
        ),
        "final_answer": LinePrefixParserNode(
            prefix="Final Answer:",
            next=[],
            is_start=True,
            is_end=True,
            field=ParserField.from_type(str),
        ),
    }
    parser = LinePrefixParser(config)
    chunks = split_string(input_str, size=25, overlap=0)
    for chunk in chunks:
        await parser.add(chunk)

    if should_throw:
        with pytest.raises(LinePrefixParserError):
            await parser.end()
    else:
        await parser.end()
        expected = "Summarize the discussion. Final Answer: The discussion thread is about ..."
        assert parser.final_state["thought"] == expected


@pytest.mark.asyncio
@pytest.mark.unit
async def test_ignores_unrelated_text_and_non_starting_nodes() -> None:
    config = {
        "first": LinePrefixParserNode(
            prefix="First:",
            field=ParserField.from_type(str),
            is_start=True,
            is_end=False,
            next=["second"],
        ),
        "second": LinePrefixParserNode(
            prefix="Second:",
            field=ParserField.from_type(str),
            is_start=False,
            is_end=True,
            next=[],
        ),
    }
    # Here we pass the config as a dict of node instances.
    parser = LinePrefixParser(config, options=LinePrefixParserOptions(wait_for_start_node=True))
    await parser.add("      Random text\nthat\nshould be ignored.\n")
    await parser.add("  Second: This should be ignored too.\n")
    await parser.add("First: first\n")
    await parser.add("Second: second\n")
    await parser.end()
    expected = {
        "first": "first",
        "second": "second\n",
    }
    assert parser.final_state == expected


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("end_on_repeat", [True, False])
async def test_ignores_other_prefixes(end_on_repeat: bool) -> None:
    config = {
        "thought": LinePrefixParserNode(
            prefix="Thought:",
            field=ParserField.from_type(str),
            is_start=True,
            is_end=True,
            next=["answer"],
        ),
        "answer": LinePrefixParserNode(
            prefix="Answer:",
            field=ParserField.from_type(str),
            is_start=True,
            is_end=True,
            next=[],
        ),
    }
    parser = LinePrefixParser(config, options=LinePrefixParserOptions(end_on_repeat=end_on_repeat))
    await parser.add("Answer: The answer is 39, see the intermediate steps that I did:\n")
    await parser.add("Thought: 3*(2+5+6)=? can be rewritten as 3*13\n")
    await parser.add("Analyze: 3*13=39\n")
    await parser.add("Outcome: 39")
    await parser.end()
    expected = {
        "answer": (
            "The answer is 39, see the intermediate steps that I did:\n"
            "Thought: 3*(2+5+6)=? can be rewritten as 3*13\n"
            "Analyze: 3*13=39\n"
            "Outcome: 39"
        ),
    }
    assert parser.final_state == expected


@pytest.mark.asyncio
@pytest.mark.unit
async def test_premature_ends_when_existing_node_visited() -> None:
    config = {
        "thought": LinePrefixParserNode(
            prefix="Thought:",
            field=ParserField.from_type(str),
            is_start=True,
            is_end=False,
            next=["tool_input", "final_answer"],
        ),
        "tool_input": LinePrefixParserNode(
            prefix="Tool Input:",
            field=ParserField.from_type(str),
            is_start=False,
            is_end=True,
            next=["tool_output"],
        ),
        "tool_output": LinePrefixParserNode(
            prefix="Tool Output:",
            field=ParserField.from_type(str),
            is_start=False,
            is_end=True,
            next=[],
        ),
        "final_answer": LinePrefixParserNode(
            prefix="Final Answer:",
            field=ParserField.from_type(str),
            is_start=False,
            is_end=True,
            next=[],
        ),
    }
    parser = LinePrefixParser(config, options=LinePrefixParserOptions(end_on_repeat=True))
    text = "\n".join(["Thought: Hello!", "Tool Input: {}", "Thought: This is ignored."])
    await parser.add(text)
    assert parser.done is True
    expected = {
        "thought": "Hello!",
        "tool_input": "{}",
    }
    assert parser.final_state == expected


@pytest.mark.unit
def test_throws_when_no_node_provided() -> None:
    with pytest.raises(ValueError):
        LinePrefixParser({})


@pytest.mark.unit
def test_throws_when_no_start_node_provided() -> None:
    with pytest.raises(ValueError) as excinfo:
        LinePrefixParser(
            {
                "a": LinePrefixParserNode(
                    prefix="A:",
                    is_start=False,
                    is_end=True,
                    next=[],
                    field=ParserField.from_type(str),
                )
            }
        )
    assert "At least one start node must be provided!" in str(excinfo.value)


@pytest.mark.unit
def test_throws_when_no_end_node_provided() -> None:
    with pytest.raises(ValueError) as excinfo:
        LinePrefixParser(
            {
                "a": LinePrefixParserNode(
                    prefix="A:",
                    is_start=True,
                    is_end=False,
                    next=[],
                    field=ParserField.from_type(str),
                )
            }
        )
    assert "At least one end node must be provided!" in str(excinfo.value)


@pytest.mark.unit
def test_throws_for_non_existing_transition() -> None:
    with pytest.raises(ValueError) as excinfo:
        LinePrefixParser(
            {
                "a": LinePrefixParserNode(
                    prefix="A:",
                    is_start=True,
                    is_end=True,
                    next=["b"],
                    field=ParserField.from_type(str),
                )
            }
        )
    assert "Node 'a' contains a transition to non-existing node" in str(excinfo.value)


@pytest.mark.unit
def test_throws_when_self_reference() -> None:
    with pytest.raises(ValueError) as excinfo:
        LinePrefixParser(
            {
                "a": LinePrefixParserNode(
                    prefix="A:",
                    is_start=True,
                    is_end=True,
                    next=["a"],
                    field=ParserField.from_type(str),
                )
            }
        )
    assert "Node 'a' cannot point to itself." in str(excinfo.value)
