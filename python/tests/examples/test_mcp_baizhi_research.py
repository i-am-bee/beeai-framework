# SPDX-License-Identifier: Apache-2.0
"""Offline tests for the authenticated Baizhi MCP example."""

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

pytest.importorskip("mcp", reason="Optional module [mcp] not installed.")
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import Tool as MCPToolInfo

from beeai_framework.tools.mcp import MCPTool
from examples.tools.mcp_baizhi_research import ALLOWED_TOOLS, select_tools, validate_endpoint

pytestmark = pytest.mark.unit


def info(name: str) -> MCPToolInfo:
    return MCPToolInfo(
        name=name,
        description=f"Synthetic {name}",
        inputSchema={"type": "object", "properties": {"query": {"type": "string"}}},
    )


def session_with_pages(*pages: tuple[list[MCPToolInfo], str | None]) -> AsyncMock:
    session = AsyncMock(spec=ClientSession)
    session.list_tools.side_effect = [SimpleNamespace(tools=tools, nextCursor=cursor) for tools, cursor in pages]
    return session


@pytest.mark.asyncio
async def test_discovery_paginates_and_preserves_tool_schemas() -> None:
    session = session_with_pages(
        ([info("websearch_search"), info("unrelated")], "page-2"),
        ([info("web_scrape"), info("web_extract")], None),
    )

    tools = await select_tools(session)

    assert [tool.name for tool in tools] == list(ALLOWED_TOOLS)
    assert tools[0].input_schema.model_json_schema()["properties"]["query"]["type"] == "string"
    assert session.list_tools.await_args_list[0].kwargs == {"cursor": None}
    assert session.list_tools.await_args_list[1].kwargs == {"cursor": "page-2"}


@pytest.mark.asyncio
async def test_native_streamable_http_session_uses_auth_and_dispatches_mcp_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = "https://selected.example/mcp"
    actual_client = httpx.AsyncClient
    requests: list[httpx.Request] = []
    tool_calls: list[dict[str, Any]] = []

    def server(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["authorization"] == "Bearer fixture-key"
        assert str(request.url) == endpoint
        message = json.loads(request.content)
        if "id" not in message:
            return httpx.Response(202)
        method = message["method"]
        if method == "initialize":
            result: dict[str, Any] = {
                "protocolVersion": message["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "synthetic", "version": "1"},
            }
        elif method == "tools/list":
            cursor = (message.get("params") or {}).get("cursor")
            result = {
                "tools": [
                    {
                        "name": name,
                        "description": f"Synthetic {name}",
                        "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}},
                    }
                    for name in (("websearch_search", "unrelated") if cursor is None else ("web_scrape", "web_extract"))
                ],
            }
            if cursor is None:
                result["nextCursor"] = "page-2"
        elif method == "tools/call":
            tool_calls.append(message["params"])
            result = {"content": [{"type": "text", "text": "synthetic result"}], "isError": False}
        else:
            raise AssertionError(method)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": result})

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(server)
        return actual_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    async with (
        httpx.AsyncClient(
            timeout=10, follow_redirects=False, trust_env=False, headers={"Authorization": "Bearer fixture-key"}
        ) as client,
        streamable_http_client(endpoint, http_client=client) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        tools = await select_tools(session)
        result = await tools[0].run({"query": "BeeAI"})

    assert result.result.text == "synthetic result"
    assert [call["name"] for call in tool_calls] == ["websearch_search"]
    assert all(request.url == endpoint for request in requests)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pages, message",
    [
        (([info("websearch_search"), info("web_scrape")], None), "missing: web_extract"),
        (
            ([info("websearch_search"), info("websearch_search"), info("web_scrape"), info("web_extract")], None),
            "duplicate: websearch_search",
        ),
        ((([info("websearch_search")], "loop"), ([info("web_scrape")], "loop")), "looping cursor"),
    ],
)
async def test_invalid_discovery_stops_before_tool_use(pages: Any, message: str) -> None:
    session = (
        session_with_pages(*pages)
        if isinstance(pages, tuple) and pages and isinstance(pages[0], tuple)
        else session_with_pages(pages)
    )

    with pytest.raises(ValueError, match=message):
        await select_tools(session)


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://example.com/mcp",
        "https://user:pass@example.com/mcp",
        "https://example.com/mcp?token=secret",
        "https://example.com/mcp#secret",
    ],
)
def test_endpoint_validation_rejects_unsafe_urls(endpoint: str) -> None:
    with pytest.raises(ValueError):
        validate_endpoint(endpoint)


def test_mcp_tools_are_framework_tools() -> None:
    assert MCPTool.__name__ == "MCPTool"
