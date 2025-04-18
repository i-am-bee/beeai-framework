import asyncio
from typing import Any, Unpack

from pydantic import BaseModel, InstanceOf

from beeai_framework.adapters.ollama import OllamaChatModel
from beeai_framework.agents.tool_calling import ToolCallingAgent
from beeai_framework.backend import AnyMessage, ChatModel, ChatModelParameters, UserMessage
from beeai_framework.context import Run, RunContext
from beeai_framework.emitter import Emitter, EmitterOptions
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.plugins.plugin import Plugin, PluginKwargs
from beeai_framework.plugins.types import PluggableInstanceRegistry, PluggableRegistry
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.weather import OpenMeteoTool
from beeai_framework.utils import ModelLike
from beeai_framework.utils.models import to_model
from beeai_framework.workflows.agent import AgentWorkflow


async def example_plugin() -> None:
    class Input(BaseModel):
        prompt: str

    class Output(BaseModel):
        messages: list[InstanceOf[AnyMessage]]

    class MyAgent(Plugin[Input, Output]):
        def __init__(self, model: ChatModel) -> None:
            super().__init__()
            self._model = model

        @property
        def emitter(self) -> Emitter:
            return Emitter.root().child(namespace=["agent", "my_agent"])

        @property
        def name(self) -> str:
            return "my_agent"

        @property
        def description(self) -> str:
            return "Agent to automatize your daily tasks so you can take a vacation."

        @property
        def input_schema(self) -> type[Input]:
            return Input

        @property
        def output_schema(self) -> type[Output]:
            return Output

        def run(self, input: ModelLike[Input], /, **kwargs: Unpack[PluginKwargs]) -> Run[Output]:
            async def handler(context: RunContext) -> Output:
                input_formatted = to_model(Input, input)
                await context.emitter.emit("before_llm_call", {"prompt": input_formatted.prompt})
                response = await self._model.create(messages=[UserMessage(input_formatted.prompt)], stream=True)
                await context.emitter.emit("after_llm_call", {"response": response.get_text_content()})
                return Output(messages=response.messages)

            return RunContext.enter(self, handler, signal=None, run_params={"input": input})

    model = ChatModel.from_name("ollama:llama3.1")
    model.config(parameters=ChatModelParameters(max_tokens=5))

    agent = MyAgent(model)

    await agent.run(agent.input_schema(prompt="Hello, how are you?")).on(
        lambda event: not event.context.get("internal"),
        lambda data, event: print(
            event.path,
            event.trace.run_id[0:8],
            event.trace.parent_run_id[0:8] if event.trace.parent_run_id else None,
            data.model_dump() if isinstance(data, BaseModel) else data,
        ),
        EmitterOptions(match_nested=True),
    )


async def workflow_as_plugin() -> None:
    llm = ChatModel.from_name("ollama:llama3.1")
    workflow = AgentWorkflow(name="Travel Advisor")
    workflow.add_agent(
        name="Weather Forecaster",
        role="A diligent weather forecaster",
        instructions="You specialize in reporting on the weather.",
        tools=[OpenMeteoTool()],
        llm=llm,
    )
    plugin = workflow.as_plugin()
    response = await plugin.run({"inputs": [UserMessage("What's the current weather in Boston?")]})
    print(response.result)


async def registry_and_autoloading() -> None:
    class PluginLoader:
        def __init__(self, *, registry: PluggableRegistry | None = None) -> None:
            self.factories = registry or PluggableRegistry()
            self.instances = PluggableInstanceRegistry()

        def load(self, /, config: dict[str, Any]) -> None:
            for name, options in config.items():
                plugin_name = options["plugin"]
                if not isinstance(plugin_name, str):
                    raise ValueError(f"Plugin name must be a string, got {type(plugin_name)}")

                plugin_parameters = options.get("parameters", {})
                if not isinstance(plugin_parameters, dict):
                    raise ValueError(f"Plugin parameters must be a dict, got {type(plugin_parameters)}")

                plugin = self.factories.lookup(plugin_name)
                instance = plugin.ref(**self._parse_plugin_parameters(plugin_parameters))
                self.instances.register(instance, name=name)

        def _parse_plugin_parameters(self, /, input: Any) -> Any:
            if isinstance(input, dict):
                return {key: self._parse_plugin_parameters(value) for key, value in input.items()}
            elif isinstance(input, list):
                return [self._parse_plugin_parameters(item) for item in input]

            if isinstance(input, str) and input.startswith("#"):
                return self.instances.lookup(input[1:]).ref
            else:
                return input

    # We start by registering the plugins we want to use
    loader = PluginLoader()

    loader.factories.register(OllamaChatModel)
    loader.factories.register(OpenMeteoTool)
    loader.factories.register(DuckDuckGoSearchTool)
    loader.factories.register(UnconstrainedMemory)
    loader.factories.register(ToolCallingAgent)

    # We retrieve the configuration from an external source (like YAML)
    user_config = {
        "ollama_granite31_chat_model": {
            "plugin": "OllamaChatModel",
            "parameters": {"model_id": "llama3.1"},
        },
        "weather_tool": {
            "plugin": "OpenMeteoTool",
            "parameters": {},
        },
        "duckduckgo_search_tool": {
            "plugin": "DuckDuckGoSearchTool",
            "parameters": {},
        },
        "unconstrained_memory": {
            "plugin": "UnconstrainedMemory",
            "parameters": {},
        },
        "my_agent": {
            "plugin": "ToolCallingAgent",
            "parameters": {
                "llm": "#ollama_granite31_chat_model",
                "tools": ["#weather_tool", "#duckduckgo_search_tool"],
                "memory": "#unconstrained_memory",
                "save_intermediate_steps": True,
            },
        },
    }

    # We convert the plain configuration into instances of the plugins
    loader.load(user_config)

    # We retrieve user actions (from an external source like API / CLI)
    user_actions = [
        {
            "plugin": "my_agent",
            "input": {"prompt": "Who is the president of Czech Republic?"},
        },
        {
            "plugin": "my_agent",
            "input": {"prompt": "How old is he?"},
        },
        {
            "plugin": "my_agent",
            "input": {"prompt": "What is the current weather in Prague?"},
        },
    ]

    for action in user_actions:
        name, input = action["plugin"], action["input"]
        assert isinstance(name, str)
        assert isinstance(input, dict)
        print(f"Running '{name}' with input {input}")

        instance = loader.instances.lookup(name).ref
        response = await instance.as_plugin().run(input)
        print("-> ", response.result)


if __name__ == "__main__":
    asyncio.run(registry_and_autoloading())
