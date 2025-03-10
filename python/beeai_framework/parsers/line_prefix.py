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

from collections.abc import Callable
from enum import StrEnum
from typing import Any, NoReturn

from pydantic import BaseModel, InstanceOf, ValidationError

from beeai_framework.emitter import Emitter
from beeai_framework.errors import FrameworkError
from beeai_framework.parsers.field import ParserField
from beeai_framework.utils.strings import trim_left_spaces

NEW_LINE_CHARACTER = "\n"


class LinePrefixParserUpdate(BaseModel):
    key: str
    value: Any
    delta: str
    field: InstanceOf[ParserField[Any]]


class LinePrefixParserLine(BaseModel):
    value: str
    new_line: bool


class LinePrefixParserExtractedLine(BaseModel):
    key: str
    value: str
    partial: bool


class LinePrefixParserOptions(BaseModel):
    fallback: Callable[[str], list[dict[str, str]]] | None = None
    end_on_repeat: bool = False
    wait_for_start_node: bool = False
    silent_nodes: list[str] | None = []


class LinePrefixParserNode(BaseModel):
    prefix: str
    next: list[str] = []
    field: InstanceOf[ParserField[Any]]
    is_start: bool = False
    is_end: bool = False


Nodes = dict[str, LinePrefixParserNode]
Customizer = Callable[[Nodes, LinePrefixParserOptions], tuple[Nodes, LinePrefixParserOptions]]


def _lines_to_string(lines: list["LinePrefixParserLine"]) -> str:
    result = ""
    for line in lines:
        if line.new_line:
            result += "\n" + line.value
        else:
            result += line.value
    return result


class LinePrefixParser:
    def __init__(self, nodes: Nodes, options: LinePrefixParserOptions | None = None) -> None:
        if options is None:
            options = LinePrefixParserOptions()
        self.nodes: Nodes = nodes
        self.options: LinePrefixParserOptions = options
        self.emitter: Emitter = Emitter(
            creator=self,
            namespace=["parser", "line"],
            events={
                "update": LinePrefixParserUpdate,
                "partial_update": LinePrefixParserUpdate,
            },
        )
        self.lines: list[LinePrefixParserLine] = []
        self.excluded_lines: list[LinePrefixParserLine] = []
        self.done: bool = False
        self.last_node_key: str | None = None

        # final_state will contain the final parsed values.
        self.final_state: dict[str, Any] = {}
        # partial_state holds incremental updates.
        self.partial_state: dict[str, str] = {}

        has_start_node = False
        has_end_node = False
        for key, node in self.nodes.items():
            has_start_node = has_start_node or node.is_start
            has_end_node = has_end_node or node.is_end
            for next_key in node.next:
                if key == next_key:
                    raise ValueError(f"Node '{key}' cannot point to itself.")
                if next_key not in self.nodes:
                    raise ValueError(f"Node '{key}' contains a transition to non-existing node '{next_key}'.")
        if not has_start_node:
            raise ValueError("At least one start node must be provided!")
        if not has_end_node:
            raise ValueError("At least one end node must be provided!")

    def fork(self, customizer: Customizer) -> "LinePrefixParser":
        new_nodes, new_options = customizer(self.nodes, self.options)
        return LinePrefixParser(new_nodes, new_options)

    async def add(self, chunk: str) -> None:
        if not chunk or self.done:
            return

        parts = chunk.split(NEW_LINE_CHARACTER)
        for i, value in enumerate(parts):
            if i == 0:
                if not self.lines:
                    self.lines.append(LinePrefixParserLine(value=value, new_line=False))
                else:
                    self.lines[-1].value += value
            else:
                self.lines.append(LinePrefixParserLine(value=value, new_line=(len(parts) > 1)))

        while self.lines:
            line = self.lines[0]
            is_last_line = len(self.lines) == 1
            last_node = self.nodes[self.last_node_key] if self.last_node_key else None
            is_termination_node = last_node and last_node.is_end and (len(last_node.next) == 0)
            parsed_line: LinePrefixParserExtractedLine | None = None
            if not (is_termination_node or (last_node and not line.new_line)):
                parsed_line = self._extract_line(line.value)

            if is_last_line and ((parsed_line is not None and parsed_line.partial) or not line.value):
                break

            self.lines.pop(0)

            if parsed_line and not parsed_line.partial:
                assert parsed_line is not None
                if last_node:
                    if parsed_line.key not in last_node.next:
                        if parsed_line.key in self.final_state and self.options.end_on_repeat and last_node.is_end:
                            await self.end()
                            return
                        self.throw_with_context(
                            f"Transition from '{self.last_node_key}' to '{parsed_line.key}' does not exist!",
                            LinePrefixParserError.Reason.InvalidTransition,
                            extra={"line": line},
                        )

                    assert self.last_node_key is not None
                    await self._emit_final_update(self.last_node_key, last_node.field)
                elif not self.nodes[parsed_line.key].is_start:
                    if not self.options.wait_for_start_node:
                        self.throw_with_context(
                            f'Parsed text line corresponds to a node "{parsed_line.key}" which is not a start node!',
                            LinePrefixParserError.Reason.NotStartNode,
                            extra={"line": line},
                        )
                    self.excluded_lines.append(line)
                    continue

                node = self.nodes[parsed_line.key]
                node.field.write(parsed_line.value)
                await self._emit_partial_update(
                    LinePrefixParserUpdate(
                        key=parsed_line.key, value=parsed_line.value, delta=parsed_line.value, field=node.field
                    )
                )
                self.last_node_key = parsed_line.key
            elif self.last_node_key:
                if not self.nodes[self.last_node_key].field.raw:
                    line.value = trim_left_spaces(line.value)
                if line.new_line:
                    line.value = NEW_LINE_CHARACTER + line.value
                node = self.nodes[self.last_node_key]
                node.field.write(line.value)
                await self._emit_partial_update(
                    LinePrefixParserUpdate(
                        key=self.last_node_key, value=node.field.get_partial(), delta=line.value, field=node.field
                    )
                )
            else:
                self.excluded_lines.append(line)

    def throw_with_context(self, message: str, reason: str, extra: dict[str, Any] | None = None) -> NoReturn:
        extra = extra or {}
        context_lines = self.lines + ([extra["line"]] if "line" in extra else [])
        context = {
            "lines": _lines_to_string(context_lines),
            "excludedLines": _lines_to_string(self.excluded_lines),
            "finalState": self.final_state,
            "partialState": self.partial_state,
        }
        full_message = f"The generated output does not adhere to the schema.\n{message}"
        raise LinePrefixParserError(full_message, reason=reason, context=context)

    async def end(self) -> dict[str, Any]:
        if self.done:
            return self.final_state

        if not self.last_node_key and self.options.fallback:
            stash = _lines_to_string(self.excluded_lines + self.lines)
            self.excluded_lines.clear()
            self.lines.clear()
            fallback_nodes = self.options.fallback(stash)
            fallback_text = NEW_LINE_CHARACTER.join(
                [self.nodes[node["key"]].prefix + node["value"] for node in fallback_nodes]
            )
            await self.add(fallback_text)

        self.done = True

        if not self.last_node_key:
            self.throw_with_context("Nothing valid has been parsed yet!", LinePrefixParserError.Reason.NoDataReceived)

        stash = _lines_to_string(self.lines)
        self.lines.clear()
        field = self.nodes[self.last_node_key].field
        if stash:
            field.write(stash)
            await self._emit_partial_update(
                LinePrefixParserUpdate(key=self.last_node_key, value=field.get_partial(), delta=stash, field=field)
            )
        await self._emit_final_update(self.last_node_key, field)
        current_node = self.nodes[self.last_node_key]
        if not current_node.is_end:
            self.throw_with_context(
                f"Node '{self.last_node_key}' is not an end node.",
                LinePrefixParserError.Reason.NotEndNode,
            )

        for node in self.nodes.values():
            node.field.end()

        return self.final_state

    async def _emit_partial_update(self, data: LinePrefixParserUpdate) -> None:
        if data.key in self.final_state:
            self.throw_with_context(
                f"Cannot update partial event for completed key '{data.key}'",
                LinePrefixParserError.Reason.AlreadyCompleted,
            )
        if data.key not in self.partial_state:
            self.partial_state[data.key] = ""
        self.partial_state[data.key] += data.delta
        if not (self.options.silent_nodes and data.key in self.options.silent_nodes):
            await self.emitter.emit("partial_update", data)

    async def _emit_final_update(self, key: str, field: ParserField[Any]) -> None:
        if key in self.final_state:
            self.throw_with_context(f"Duplicated key '{key}'", LinePrefixParserError.Reason.AlreadyCompleted)
        try:
            value = field.get()
            self.final_state[key] = value.model_dump()
            if not (self.options.silent_nodes and key in self.options.silent_nodes):
                await self.emitter.emit("update", LinePrefixParserUpdate(key=key, value=value, field=field, delta=""))
        except ValidationError:
            self.throw_with_context(
                f"Value for '{key}' cannot be retrieved because its value does not adhere to the appropriate schema.",
                LinePrefixParserError.Reason.InvalidSchema,
            )

    @property
    def _normalized_nodes(self) -> list[tuple[str, dict[str, Any]]]:
        sorted_nodes = sorted(self.nodes.items(), key=lambda item: len(item[1].prefix))
        return [(key, {"lowerCasePrefix": node.prefix.lower(), "ref": node}) for key, node in sorted_nodes]

    def _extract_line(self, line: str) -> LinePrefixParserExtractedLine | None:
        trimmed_line = trim_left_spaces(line)
        if not trimmed_line:
            return None

        for key, node_info in self._normalized_nodes:
            lower_case_prefix = node_info["lowerCasePrefix"]
            partial = len(lower_case_prefix) > len(trimmed_line)
            if partial:
                a, b = lower_case_prefix, trimmed_line
            else:
                a, b = trimmed_line, lower_case_prefix

            if a.lower().startswith(b.lower()):
                value = trimmed_line if partial else trim_left_spaces(trimmed_line[len(b) :])
                return LinePrefixParserExtractedLine(key=key, value=value, partial=partial)

        return None


class LinePrefixParserError(FrameworkError):
    class Reason(StrEnum):
        NoDataReceived = "NoDataReceived"
        InvalidTransition = "InvalidTransition"
        NotStartNode = "NotStartNode"
        NotEndNode = "NotEndNode"
        AlreadyCompleted = "AlreadyCompleted"
        InvalidSchema = "InvalidSchema"

    def __init__(self, message: str, *, context: dict[str, Any], reason: str) -> None:
        super().__init__(message, is_fatal=True, is_retryable=True)
        self.context = context
        self.reason = reason
