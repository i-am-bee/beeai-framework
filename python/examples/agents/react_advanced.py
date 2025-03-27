import asyncio
import sys
import traceback
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel

from beeai_framework import UnconstrainedMemory
from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.agents.react.agent import ReActAgent
from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.cancellation import AbortSignal
from beeai_framework.emitter.emitter import Emitter, EventMeta
from beeai_framework.emitter.types import EmitterOptions
from beeai_framework.errors import FrameworkError
from beeai_framework.template import PromptTemplateInput
from beeai_framework.tools.search import DuckDuckGoSearchTool
from beeai_framework.tools.tool import AnyTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from examples.helpers.io import ConsoleReader

# Load environment variables
load_dotenv()

reader = ConsoleReader()


def user_customizer(config: PromptTemplateInput[Any]) -> PromptTemplateInput[Any]:
    class UserSchema(BaseModel):
        input: str

    new_config = config.model_copy()
    new_config.input_schema = UserSchema
    new_config.template = """User: {{input}}"""
    return new_config


def no_result_customizer(config: PromptTemplateInput[Any]) -> PromptTemplateInput[Any]:
    new_config = config.model_copy()
    config.template += """\nPlease reformat your input."""
    return new_config


def not_found_customizer(config: PromptTemplateInput[Any]) -> PromptTemplateInput[Any]:
    class ToolSchema(BaseModel):
        name: str

    class NotFoundSchema(BaseModel):
        tools: list[ToolSchema]

    new_config = config.model_copy()
    new_config.input_schema = NotFoundSchema
    new_config.template = """Tool does not exist!
{{#tools.length}}
Use one of the following tools: {{#trim}}{{#tools}}{{name}},{{/tools}}{{/trim}}
{{/tools.length}}"""
    return new_config


def create_agent() -> ReActAgent:
    """Create and configure the agent with tools and LLM"""

    llm = OllamaChatModel("llama3.1")

    templates: dict[str, Any] = {
        "user": lambda template: template.fork(customizer=user_customizer),
        "system": lambda template: template.update(
            defaults={"instructions": "You are a helpful assistant that uses tools to answer questions."}
        ),
        "tool_no_result_error": lambda template: template.fork(customizer=no_result_customizer),
        "tool_not_found_error": lambda template: template.fork(customizer=not_found_customizer),
    }

    tools: list[AnyTool] = [
        # WikipediaTool(),
        OpenMeteoTool(),
        DuckDuckGoSearchTool(),
    ]

    agent = ReActAgent(llm=llm, templates=templates, tools=tools, memory=UnconstrainedMemory())

    return agent


def process_agent_events(data: Any, event: EventMeta) -> None:
    """Process agent events and log appropriately"""

    if event.name == "error":
        reader.write("Agent 🤖 : ", FrameworkError.ensure(data.error).explain())
    elif event.name == "retry":
        reader.write("Agent 🤖 : ", "retrying the action...")
    elif event.name == "update":
        reader.write(f"Agent({data.update.key}) 🤖 : ", data.update.parsed_value)
    elif event.name == "start":
        reader.write("Agent 🤖 : ", "starting new iteration")
    elif event.name == "success":
        reader.write("Agent 🤖 : ", "success")


def observer(emitter: Emitter) -> None:
    emitter.on("*", process_agent_events, EmitterOptions(match_nested=False))


async def main() -> None:
    """Main application loop"""

    agent = create_agent()

    for prompt in reader:
        response = await agent.run(
            prompt=prompt,
            execution=AgentExecutionConfig(max_retries_per_step=3, total_max_retries=10, max_iterations=20),
            signal=AbortSignal.timeout(2 * 60 * 1000),
        ).observe(observer)

        reader.write("Agent 🤖 : ", response.result.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
