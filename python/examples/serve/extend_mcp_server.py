from typing import Any

from mcp.server.fastmcp.prompts.base import Prompt as MCPPrompt
from mcp.server.fastmcp.prompts.base import PromptArgument
from pydantic import BaseModel

from beeai_framework.adapters.mcp import MCPServer, MCPServerConfig
from beeai_framework.template import PromptTemplate


def add_prompt_template_factory() -> None:
    def factory(instance: PromptTemplate[Any]) -> MCPPrompt:
        return MCPPrompt(
            name=instance.name,
            title=instance.name,
            description=instance.description,
            arguments=[
                PromptArgument(
                    name=k, description=v.description, required=v.default is None and v.default_factory is None
                )
                for k, v in instance.input_schema.model_fields.items()
            ],
            fn=lambda **kwargs: instance.render(kwargs),
        )

    MCPServer.register_factory(PromptTemplate, factory, override=True)


def run_server() -> None:
    class GreetingTemplateModel(BaseModel):
        name: str

    my_template = PromptTemplate(
        template="Hello {name}",
        schema=GreetingTemplateModel,
    )

    server = MCPServer(config=MCPServerConfig(transport="streamable-http"))
    server.register(my_template)
    server.serve()


if __name__ == "__main__":
    add_prompt_template_factory()
    run_server()
