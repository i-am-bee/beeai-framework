import asyncio
import os
import sys
import traceback

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from beeai_framework.agents.react.agent import ReActAgent
from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import SystemMessage, UserMessage
from beeai_framework.emitter.emitter import EventMeta
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.token_memory import TokenMemory
from beeai_framework.tools.mcp_tools import MCPTool
from examples.helpers.io import ConsoleReader

load_dotenv()

server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-github"],
    env={
        "GITHUB_TOKEN": os.environ["GITHUB_TOKEN"],
        "GITHUB_OWNER": os.environ["GITHUB_OWNER"],
        "GITHUB_REPO": os.environ["GITHUB_REPO"],
        "PATH": os.getenv("PATH", default=""),
    },
)


async def setup_github_tools() -> list[MCPTool]:
    async with stdio_client(server_params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        github_tools = await MCPTool.from_client(session)
        print(f"Available GitHub tools: {[tool.name for tool in github_tools]}")
        return github_tools


async def create_issue_agent() -> ReActAgent:
    tools = await setup_github_tools()
    
    llm = ChatModel.from_name("ollama:llama3.1")
    
    memory = TokenMemory(llm)
    await memory.add(SystemMessage(f"""
You are a GitHub Issue Manager assistant for the repository {os.environ.get('GITHUB_OWNER')}/{os.environ.get('GITHUB_REPO')}.

You have access to GitHub tools that can list, create, and manage issues. When helping users:
1. Always use the appropriate GitHub tool instead of explaining GitHub concepts
2. Use the exact repository owner and name in your tool calls
3. For listing issues, use the list_issues tool with the proper state parameter
4. Be concise and focus on executing the user's requests

When listing issues, include specific issue numbers, titles, and states in your response.
"""))
    
    agent = ReActAgent(
        llm=llm,
        tools=tools,
        memory=memory
    )
    
    return agent


def print_tool_usage(data: any, event: EventMeta) -> None:
    if event.name == "update":
        if data.update.key == "tool_name":
            print(f"Using tool: {data.update.parsed_value}")
        elif data.update.key == "tool_input":
            print(f"Tool input: {data.update.parsed_value}")
    elif event.name == "error":
        print(f"Error: {FrameworkError.ensure(data.error).explain()}")


async def main() -> None:
    reader = ConsoleReader()
    
    print(f"GitHub Issue Manager for {os.environ.get('GITHUB_OWNER')}/{os.environ.get('GITHUB_REPO')}")
    print("Type 'exit' to quit.")
    print("-" * 50)
    
    agent = await create_issue_agent()
    
    for prompt in reader:
        if prompt.lower() == "exit":
            print("Goodbye!")
            break
        
        response = await agent.run(
            prompt=prompt,
            execution=AgentExecutionConfig(
                max_retries_per_step=3,
                max_iterations=10
            )
        ).on("*", print_tool_usage)
        
        reader.write("GitHub Issue Manager: ", response.result.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
