# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
"""Run a bounded research task with Baizhi's hosted MCP tools.

Set ``BAIZHI_API_KEY`` before running this opt-in example. The key is sent only
as a Bearer header to the configured HTTPS endpoint and is never printed.
"""

import asyncio
import os
from collections import Counter
from urllib.parse import urlparse

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import ListToolsResult
from mcp.types import Tool as MCPToolInfo

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.requirements.conditional import ConditionalRequirement
from beeai_framework.backend import ChatModel
from beeai_framework.tools.mcp import MCPTool

BAIZHI_MCP_ENDPOINT = "https://agent-toolkit.app.baizhi.cloud/mcp"
ALLOWED_TOOLS = ("websearch_search", "web_scrape", "web_extract")


def validate_endpoint(endpoint: str) -> str:
    """Reject endpoints that could accidentally disclose credentials."""

    parsed = urlparse(endpoint)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("BAIZHI_MCP_ENDPOINT must be an HTTPS URL without userinfo")
    if parsed.query or parsed.fragment:
        raise ValueError("BAIZHI_MCP_ENDPOINT must not contain a query or fragment")
    return endpoint


async def discover_tools(session: ClientSession) -> list[MCPToolInfo]:
    """Fetch every MCP tools/list page and reject looping cursors."""

    tools: list[MCPToolInfo] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()

    while True:
        result: ListToolsResult = await session.list_tools(cursor=cursor)
        tools.extend(result.tools)
        next_cursor = result.nextCursor
        if not next_cursor:
            return tools
        if next_cursor in seen_cursors:
            raise ValueError("MCP tools/list returned a looping cursor")
        seen_cursors.add(next_cursor)
        cursor = next_cursor


async def select_tools(session: ClientSession) -> list[MCPTool]:
    """Discover and wrap exactly one instance of each allowed research tool."""

    discovered = await discover_tools(session)
    counts = Counter(tool.name for tool in discovered)
    missing = [name for name in ALLOWED_TOOLS if counts[name] == 0]
    duplicates = [name for name in ALLOWED_TOOLS if counts[name] > 1]
    if missing or duplicates:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if duplicates:
            details.append(f"duplicate: {', '.join(duplicates)}")
        raise ValueError("Baizhi MCP tool allowlist is incomplete (" + "; ".join(details) + ")")

    by_name = {tool.name: MCPTool(session=session, tool=tool) for tool in discovered if tool.name in ALLOWED_TOOLS}
    return [by_name[name] for name in ALLOWED_TOOLS]


async def run_research(
    task: str,
    *,
    api_key: str,
    endpoint: str = BAIZHI_MCP_ENDPOINT,
    model_name: str = "ollama:granite4:micro",
) -> str:
    """Keep the HTTP client and MCP session alive for the complete agent run."""

    if not api_key:
        raise ValueError("Set BAIZHI_API_KEY before running this example")
    endpoint = validate_endpoint(endpoint)

    async with (
        httpx.AsyncClient(
            timeout=60,
            follow_redirects=False,
            trust_env=False,
            headers={"Authorization": f"Bearer {api_key}"},
        ) as http_client,
        streamable_http_client(endpoint, http_client=http_client) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        tools = await select_tools(session)
        requirements = [
            ConditionalRequirement(tool, max_invocations=1, only_success_invocations=False) for tool in tools
        ]
        agent = RequirementAgent(
            llm=ChatModel.from_name(model_name),
            tools=tools,
            requirements=requirements,
            instructions=[
                "Use the research tools only when they add evidence to the answer.",
                "Preserve source URLs and clearly state evidence gaps or uncertainty.",
            ],
        )
        response = await agent.run(task)
        return response.last_message.text


async def main() -> None:
    api_key = os.getenv("BAIZHI_API_KEY", "")
    endpoint = os.getenv("BAIZHI_MCP_ENDPOINT", BAIZHI_MCP_ENDPOINT)
    model_name = os.getenv("BAIZHI_MODEL", "ollama:granite4:micro")
    task = os.getenv("BAIZHI_RESEARCH_TASK", "Summarize the latest BeeAI Framework release and cite sources.")
    print(await run_research(task, api_key=api_key, endpoint=endpoint, model_name=model_name))


if __name__ == "__main__":
    asyncio.run(main())
