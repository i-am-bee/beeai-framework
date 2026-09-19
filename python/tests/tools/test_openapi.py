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
