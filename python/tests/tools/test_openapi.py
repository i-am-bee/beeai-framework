# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import httpx
import pytest

from beeai_framework.tools.openapi import OpenAPITool

YAML_SPEC = """
openapi: 3.0.0
info:
  title: Demo
  version: "1.0"
servers:
  - url: https://example.invalid
paths:
  /ping:
    get:
      operationId: ping
      responses:
        "200":
          description: ok
"""


class _FakeResponse:
    def __init__(self, text: str, content_type: str) -> None:
        self.text = text
        self.headers = {"Content-Type": content_type}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> None:
        raise AssertionError("json() must not be called for a YAML response")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_from_url_parses_yaml_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """A server advertising a YAML content type must be parsed into the OpenAPI schema dict,
    not into a generator of low-level parser events (`yaml.parse` is the wrong function --
    `yaml.safe_load` is the one that returns data)."""
    fake_response = _FakeResponse(YAML_SPEC, "application/yaml; charset=utf-8")

    async def fake_get(self: httpx.AsyncClient, url: str, *args: object, **kwargs: object) -> _FakeResponse:
        return fake_response

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    tools = await OpenAPITool.from_url("https://example.invalid/openapi.yaml")

    assert len(tools) == 1
    assert tools[0].name == "ping"
    assert tools[0].url == "https://example.invalid"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unset_optional_query_parameters_are_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    schema = {
        "openapi": "3.0.0",
        "info": {"title": "Pets", "version": "1.0"},
        "servers": [{"url": "https://example.invalid"}],
        "paths": {
            "/pets": {
                "get": {
                    "operationId": "listPets",
                    "parameters": [
                        {"name": "species", "in": "query", "required": True, "schema": {"type": "string"}},
                        {"name": "limit", "in": "query", "schema": {"type": "integer"}},
                    ],
                    "responses": {"200": {"description": "ok"}},
                }
            }
        },
    }
    urls: list[str] = []

    async def fake_request(self: httpx.AsyncClient, method: str, url: str, **kwargs: object) -> httpx.Response:
        urls.append(url)
        return httpx.Response(200, text="[]")

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)

    [tool] = OpenAPITool.from_schema(schema)
    await tool.run({"parameters": {"species": "cat"}})
    await tool.run({"parameters": {"species": "cat", "limit": 0}})

    assert urls == [
        "https://example.invalid/pets?species=cat",
        "https://example.invalid/pets?species=cat&limit=0",
    ]


def _schema(server_url: str, path: str, method_spec: dict) -> dict:
    return {
        "openapi": "3.0.0",
        "info": {"title": "Pets", "version": "1.0"},
        "servers": [{"url": server_url}],
        "paths": {path: {"get": {"operationId": "op", "responses": {"200": {"description": "ok"}}, **method_spec}}},
    }


async def _run_and_capture(
    monkeypatch: pytest.MonkeyPatch, schema: dict, parameters: dict
) -> tuple[str, dict[str, str]]:
    captured: dict[str, object] = {}

    async def fake_request(self: httpx.AsyncClient, method: str, url: str, **kwargs: object) -> httpx.Response:
        captured["url"] = url
        captured["headers"] = kwargs["headers"]
        return httpx.Response(200, text="{}")

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)

    [tool] = OpenAPITool.from_schema(schema)
    await tool.run({"parameters": parameters})
    return str(captured["url"]), dict(captured["headers"])  # type: ignore[call-overload]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_path_parameters_are_substituted_not_sent_as_query(monkeypatch: pytest.MonkeyPatch) -> None:
    schema = _schema(
        "https://example.invalid",
        "/pets/{petId}",
        {"parameters": [{"name": "petId", "in": "path", "required": True, "schema": {"type": "string"}}]},
    )

    url, _ = await _run_and_capture(monkeypatch, schema, {"petId": "a/b c"})

    assert url == "https://example.invalid/pets/a%2Fb%20c"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_header_and_cookie_parameters_are_sent_as_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    schema = _schema(
        "https://example.invalid",
        "/pets",
        {
            "parameters": [
                {"name": "limit", "in": "query", "schema": {"type": "integer"}},
                {"name": "X-Request-Id", "in": "header", "schema": {"type": "string"}},
                {"name": "session", "in": "cookie", "schema": {"type": "string"}},
                {"name": "theme", "in": "cookie", "schema": {"type": "string"}},
            ]
        },
    )

    url, headers = await _run_and_capture(
        monkeypatch, schema, {"limit": 5, "X-Request-Id": "abc", "session": "s1", "theme": "dark"}
    )

    assert url == "https://example.invalid/pets?limit=5"
    assert headers["X-Request-Id"] == "abc"
    assert headers["Cookie"] == "session=s1; theme=dark"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_boolean_query_parameters_are_lowercase(monkeypatch: pytest.MonkeyPatch) -> None:
    schema = _schema(
        "https://example.invalid",
        "/pets",
        {"parameters": [{"name": "flag", "in": "query", "schema": {"type": "boolean"}}]},
    )

    url, _ = await _run_and_capture(monkeypatch, schema, {"flag": False})

    assert url == "https://example.invalid/pets?flag=false"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_server_url_path_and_query_are_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    schema = _schema(
        "https://example.invalid/api?key=k",
        "/pets/{petId}",
        {
            "parameters": [
                {"name": "petId", "in": "path", "required": True, "schema": {"type": "integer"}},
                {"name": "verbose", "in": "query", "schema": {"type": "boolean"}},
            ]
        },
    )

    url, _ = await _run_and_capture(monkeypatch, schema, {"petId": 1, "verbose": True})

    assert url == "https://example.invalid/api/pets/1?key=k&verbose=true"
