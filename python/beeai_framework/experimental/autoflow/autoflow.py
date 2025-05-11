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

import asyncio
import json
import random
import string
from typing import Any, Generic, Literal, Self, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, Field, InstanceOf

from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import (
    AssistantMessage,
    MessageToolCallContent,
    MessageToolResultContent,
    SystemMessage,
    ToolMessage,
)
from beeai_framework.errors import FrameworkError
from beeai_framework.experimental.autoflow.prompts import (
    AutoFlowTemplates,
)
from beeai_framework.experimental.autoflow.shared import (
    Runnable,
    RunnableInput,
    runnable,
    runnable_custom,
    runnable_to_tool,
)
from beeai_framework.memory import BaseMemory, UnconstrainedMemory
from beeai_framework.tools import JSONToolOutput, ToolError, ToolOutput
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.tool import AnyTool
from beeai_framework.utils.counter import RetryCounter
from beeai_framework.utils.models import to_model
from beeai_framework.utils.strings import to_json
from beeai_framework.workflows import WorkflowError

TInput = TypeVar("TInput", bound=BaseModel)
TOutput = TypeVar("TOutput", bound=BaseModel)


class AutoFlowState(BaseModel, Generic[TInput, TOutput]):
    context: dict[str, Any]
    input: TInput
    final_response: TOutput | None
    memory: InstanceOf[BaseMemory]


class Autoflow(Runnable[TInput, TOutput]):
    def __init__(
        self,
        /,
        *,
        model: ChatModel,
        name: str,
        description: str,
        input_schema: type[TInput],
        output_schema: type[TOutput],
        templates: AutoFlowTemplates | None = None,
        memory: BaseMemory | None = None,
    ) -> None:
        self.name = name
        self.memory = memory or UnconstrainedMemory()
        self.description = description
        self.input_schema = input_schema
        self.output_schema = output_schema
        self._model = model
        self._services: list[Runnable[Any, Any]] = []
        self._templates: AutoFlowTemplates = templates or AutoFlowTemplates()

    async def run(self, *args: None, **kwargs: Any) -> TOutput:
        if args:
            raise FrameworkError("Args parameters are not allowed!")

        @runnable_custom(input_schema=self.output_schema, output_schema=self.output_schema)
        async def respond_to_user(message: TOutput) -> TOutput:
            """Sends message to the user explaining what the outcome is."""

            state.final_response = to_model(self.output_schema, message)
            return message

        global_retries_counter = RetryCounter(error_type=WorkflowError, max_retries=10)
        tools = [runnable_to_tool(service) for service in [*self._services, respond_to_user]]

        state = AutoFlowState[TInput, TOutput](
            context=kwargs.pop("context", {}).copy(),
            input=to_model(self.input_schema, kwargs),
            final_response=None,
            memory=UnconstrainedMemory(),
        )

        # dummy init
        input_id = "".join(random.choice(string.ascii_letters) for _ in range(4))

        # print(state.memory.messages[0].text)
        # if kwargs:
        #    await state.memory.add(UserMessage(to_json(kwargs)))

        while state.final_response is None:
            tool_choice: Literal["required"] | AnyTool = "required" if len(tools) > 1 else tools[0]
            prefix_messages = [
                SystemMessage(
                    self._templates.system.render(
                        context=to_json(state.context) if state.context else None,
                        tools=[
                            {
                                "name": t.name,
                                "description": t.description,
                                "input_schema": to_json(t.input_schema.model_json_schema()),
                            }
                            for t in tools
                        ],
                    )
                ),
                *self.memory.messages,
                AssistantMessage(MessageToolCallContent(id=input_id, tool_name="retrieve_task", args="{}")),
                ToolMessage(
                    MessageToolResultContent(
                        tool_call_id=input_id,
                        tool_name="retrieve_task",
                        result=to_json(state.input.model_dump(exclude_none=True)),  # TODO
                    )
                ),
            ]

            response = await self._model.create(
                messages=[*prefix_messages, *state.memory.messages],
                tools=tools,
                tool_choice=tool_choice,
                use_structured_outputs_for_tools=True,
            )
            await state.memory.add_many(response.messages)

            tool_calls = response.get_tool_calls()
            for tool_call in tool_calls:
                try:
                    tool: AnyTool | None = next((tool for tool in tools if tool.name == tool_call.tool_name), None)
                    if tool is None:
                        raise ToolError(f"No tool called {tool_call.tool_name} exists!")

                    print(f"🛠️ Tool '{tool_call.tool_name}' with", json.loads(tool_call.args))
                    raw_tool_output = await tool.run(json.loads(tool_call.args))
                    tool_output: ToolOutput = (
                        raw_tool_output if isinstance(raw_tool_output, ToolOutput) else JSONToolOutput(raw_tool_output)
                    )
                    print(f"✅️ -> {tool_output.get_text_content()}")
                    await state.memory.add(
                        ToolMessage(
                            MessageToolResultContent(
                                result=tool_output.get_text_content(),
                                tool_name=tool_call.tool_name,
                                tool_call_id=tool_call.id,
                            )
                        )
                    )
                except ToolError as e:
                    err_msg = self._templates.tool_error.render({"reason": e.explain()})
                    print(f"💥️ -> {err_msg}")
                    global_retries_counter.use(e)
                    await state.memory.add(
                        ToolMessage(
                            MessageToolResultContent(
                                result=err_msg,
                                tool_name=tool_call.tool_name,
                                tool_call_id=tool_call.id,
                            )
                        )
                    )

            text_messages = response.get_text_messages()
            if not tool_calls and not text_messages:
                print("⚠️ generated empty response!")
                await state.memory.add(AssistantMessage("\n", {"tempMessage": True}))
            else:
                await state.memory.delete_many(
                    [msg for msg in state.memory.messages if msg.meta.get("tempMessage", False)]
                )

            for text_msg in text_messages:
                print("💬", text_msg.text)
                # text_msg.meta.update({"tempMessage": True})
                # await state.memory.delete(text_msg)  # TODO: remove?

            if not tool_calls:
                print("💥️ -> model did not use any tool calls!")
                # await state.memory.delete_many(response.messages)
                await state.memory.add(
                    AssistantMessage(
                        f"I will now use {respond_to_user.name} function to send the message to the user.\n"
                    ),
                )
                # await state.memory.add(
                #    AssistantMessage(
                #        f"I know the final answer. I will use '{sends_to_manager.name}' function to report to my user."
                #    ),
                # )
                # await state.memory.add(
                #     AssistantMessage(
                #         self._templates.no_tool_call.render(
                #             {
                #                 "tools": ",".join([tool.name for tool in tools]),
                #             }
                #         ),
                #         {"tempMessage": True},
                #     )
                # )

        await self.memory.add_many(state.memory.messages)

        assert state.final_response is not None
        return state.final_response

        # final_response = await self._model.create_structure(
        #     schema=self.output_schema,
        #     messages=[
        #         SystemMessage(
        #             self._templates.resolver.render(
        #                 AutoFlowResolverPromptInput(
        #                     expected_output=to_json(self.output_schema.model_json_schema(mode="serialization"))
        #                 )
        #             )
        #         ),
        #         *memory.messages[1:],
        #     ],
        # )
        # parsed: TOutput = to_model(self.output_schema, final_response.object)
        # return parsed

    def register(self, target: RunnableInput[Any]) -> Self:
        service = runnable(target)

        if service not in self._services:
            self._services.append(service)

        return self


load_dotenv()


async def main() -> None:
    class Input(BaseModel):
        task: str = Field(description="The user defined task.")

    class Output(BaseModel):
        final_answer: str = Field(description="The final answer to the question.")

    flow = Autoflow(
        model=ChatModel.from_name("ollama:llama3.1", use_openai=False),
        # model=ChatModel.from_name("watsonx:meta-llama/llama-3-1-8b-instruct"),
        name="test",
        description="test",
        input_schema=Input,
        output_schema=Output,
    )

    flow.register(DuckDuckGoSearchTool(max_results=3))

    @flow.register
    async def ask_for_clarification(question: str) -> str:
        """Asks the manager for more clarifying questions."""

        return "Czech Republic"

        # reader = ConsoleReader()
        # response: str = reader.ask_single_question(f"🤖{question}\n")
        # return response

    for prompt in ["What is the name of the president and president's wife in my favorite country?"]:
        print("👤", prompt)
        response = await flow.run(task=prompt)
        print(response)


if __name__ == "__main__":
    LiteLLMChatModel.litellm_debug(False)
    asyncio.run(main())
