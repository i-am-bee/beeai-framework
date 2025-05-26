# Copyright 2025 © BeeAI a Series of LF Projects, LLC
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

import inspect
import os
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from functools import cached_property
from typing import Any, ClassVar, TypeVar, overload

import chevron
from pydantic import BaseModel
from typing_extensions import Unpack

from beeai_framework.context import Run, RunContext, storage
from beeai_framework.emitter import Emitter
from beeai_framework.plugins.plugin import AnyPlugin, Plugin, PluginKwargs
from beeai_framework.utils.lists import cast_list
from beeai_framework.utils.models import ModelLike, get_input_schema, get_output_schema, to_model
from beeai_framework.utils.strings import to_safe_word

TInput = TypeVar("TInput", bound=BaseModel)
TOutput = TypeVar("TOutput", bound=BaseModel)

TFunction = Callable[..., Any]


@overload
def plugin(
    plugin_function: TFunction,
    /,
    *,
    name: str | None = ...,
    description: str | None = ...,
    input_schema: type[TInput] | None = ...,
    output_schema: type[TOutput] | None = ...,
    emitter: Emitter | None = None,
) -> AnyPlugin: ...
@overload
def plugin(
    *,
    name: str | None = ...,
    description: str | None = ...,
    input_schema: type[TInput] | None = ...,
    output_schema: type[TOutput] | None = ...,
    emitter: Emitter | None = None,
) -> Callable[[TFunction], AnyPlugin]: ...
def plugin(
    fn: TFunction | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    input_schema: type[TInput] | None = None,
    output_schema: type[TOutput] | None = None,
    emitter: Emitter | None = None,
) -> AnyPlugin | Callable[[TFunction], AnyPlugin]:
    def create_plugin(fn: TFunction) -> AnyPlugin:
        plugin_name = name or fn.__name__
        plugin_description = description or inspect.getdoc(fn)
        plugin_input: type[TInput] = input_schema or get_input_schema(fn)
        plugin_output: type[TOutput] = output_schema or get_output_schema(fn)

        if plugin_description is None:
            raise ValueError("No plugin description provided.")

        class FunctionPlugin(Plugin[TInput, TOutput]):
            _auto_register: ClassVar[bool] = False

            name = plugin_name
            description = plugin_description or ""
            input_schema = plugin_input
            output_schema = plugin_output

            @cached_property
            def emitter(self) -> Emitter:
                if emitter is not None:
                    return emitter

                return Emitter.root().child(
                    namespace=["plugin", "custom", to_safe_word(self.name)],
                    creator=self,
                )

            def run(self, input: ModelLike[TInput], /, **kwargs: Unpack[PluginKwargs]) -> Run[TOutput]:
                async def handler(context: RunContext) -> TOutput:
                    target_input = to_model(self.input_schema, input).model_dump()
                    if inspect.iscoroutinefunction(fn):
                        result = await fn(**target_input)
                    else:
                        result = fn(**target_input)

                    return to_model(plugin_output, result)

                return RunContext.enter(self, handler, signal=kwargs.get("signal"), run_params={"input": input})

        return FunctionPlugin()

    if fn is None:
        return create_plugin
    else:
        return create_plugin(fn)


def render_env_variables(template: str | dict[Any, Any] | list[Any]) -> str | dict[Any, Any] | list[Any]:
    """Renders a template with environment variables.
    Args:
        template: a chevron template.
    """
    if isinstance(template, str):
        return chevron.render(template=template, data=dict(os.environ))
    elif isinstance(template, dict):
        for key, value in template.items():
            if isinstance(value, str | dict | list):
                template[key] = render_env_variables(value)
        return template
    elif isinstance(template, list):
        for i, value in enumerate(template):
            if isinstance(value, str | dict | list):
                template[i] = render_env_variables(value)
        return template
    return template


def transfer_run_context(source: RunContext | None = None) -> Callable[[RunContext], None]:
    source_context = source or storage.get()

    def transfer_middleware(target: RunContext) -> None:
        target.run_id = source_context.run_id
        target.parent_id = source_context.parent_id
        target.group_id = source_context.group_id
        target.emitter.trace = source_context.emitter.trace
        target.emitter.namespace = source_context.emitter.namespace

    return transfer_middleware


T = TypeVar("T")


def topological_sort(items: Mapping[str, T], attr_name: str) -> list[tuple[str, T]]:
    # Build the graph and in-degree count
    graph = defaultdict(list)
    in_degree = {key: 0 for key in items}

    for key, value in items.items():
        based_on = cast_list(getattr(value, attr_name) or [])
        for dep in based_on:
            if dep not in items:
                continue

            graph[dep].append(key)
            in_degree[key] += 1

    # Queue for items with no dependencies
    queue = deque([k for k, v in in_degree.items() if v == 0])
    sorted_list = []

    while queue:
        node = queue.popleft()
        sorted_list.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(sorted_list) != len(items):
        raise ValueError("Cycle detected or missing dependencies!")

    return [(key, items[key]) for key in sorted_list]
