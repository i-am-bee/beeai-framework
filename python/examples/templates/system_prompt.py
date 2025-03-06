import sys
import traceback

from beeai_framework.agents.react.runners.default.prompts import (
    SystemPromptTemplate,
    ToolDefinition,
)
from beeai_framework.errors import FrameworkError
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from beeai_framework.utils.strings import to_json


def main() -> None:
    tool = OpenMeteoTool()

    tool_def = ToolDefinition(
        name=tool.name,
        description=tool.description,
        input_schema=to_json(tool.input_schema.model_json_schema()),
    )

    # Render the granite system prompt
    prompt = SystemPromptTemplate.render(
        instructions="You are a helpful AI assistant!", tools=[tool_def], tools_length=1
    )

    print(prompt)


if __name__ == "__main__":
    try:
        main()
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
