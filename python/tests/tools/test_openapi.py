# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any
from urllib.parse import parse_qs

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


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content_type", "expected_body"),
    [
        ("application/json", {"name": "Bee", "tags": ["a", "b"]}),
        ("application/x-www-form-urlencoded", {"name": ["Bee"], "tags": ["a", "b"]}),
    ],
)
async def test_request_body_is_encoded_per_content_type(
    monkeypatch: pytest.MonkeyPatch, content_type: str, expected_body: dict[str, Any]
) -> None:
    schema = {
        "openapi": "3.0.0",
        "info": {"title": "Pets", "version": "1.0"},
        "servers": [{"url": "https://example.invalid"}],
        "paths": {
            "/pets": {
                "post": {
                    "operationId": "addPet",
                    "requestBody": {
                        "required": True,
                        "content": {
                            content_type: {
                                "schema": {
                                    "type": "object",
                                    "required": ["name", "tags"],
                                    "properties": {
                                        "name": {"type": "string"},
                                        "tags": {"type": "array", "items": {"type": "string"}},
                                    },
                                }
                            }
                        },
                    },
                    "responses": {"200": {"description": "ok"}},
                }
            }
        },
    }
    requests: list[httpx.Request] = []

    async def fake_send(self: httpx.AsyncClient, request: httpx.Request, **kwargs: object) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, text="{}")

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)

    [tool] = OpenAPITool.from_schema(schema)
    await tool.run({"body": {"name": "Bee", "tags": ["a", "b"]}})

    [request] = requests
    assert request.headers["Content-Type"] == content_type
    if content_type == "application/json":
        assert json.loads(request.content) == expected_body
    else:
        assert parse_qs(request.content.decode()) == expected_body
