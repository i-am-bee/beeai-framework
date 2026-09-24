# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from typing import Unpack

import pytest

from beeai_framework.adapters.openai.serve import server as server_mod
from beeai_framework.adapters.openai.serve.server import OpenAIServer, OpenAIServerConfig, register
from beeai_framework.backend.message import AnyMessage, AssistantMessage
from beeai_framework.emitter import Emitter
from beeai_framework.runnable import Runnable, RunnableOptions, RunnableOutput, runnable_entry


class DummyRunnable(Runnable[RunnableOutput]):
    @property
    def emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["dummy"])

    @runnable_entry
    async def run(self, input: list[AnyMessage], /, **kwargs: Unpack[RunnableOptions]) -> RunnableOutput:
        return RunnableOutput(output=[AssistantMessage(content="pong")])


@pytest.fixture
def captured_warnings(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Stub out uvicorn.run and capture warnings emitted during serve()."""
    register()
    monkeypatch.setattr(server_mod.uvicorn, "run", lambda *args, **kwargs: None)
    warnings: list[str] = []
    monkeypatch.setattr(server_mod.logger, "warning", lambda msg, *a, **k: warnings.append(str(msg)))
    yield warnings


def _serve(config: OpenAIServerConfig) -> None:
    server = OpenAIServer(config=config)
    server.register(DummyRunnable(), name="dummy-model")
    server.serve()


def _has_insecure_warning(warnings: list[str]) -> bool:
    return any("without authentication" in w for w in warnings)


@pytest.mark.unit
def test_default_host_is_loopback() -> None:
    config = OpenAIServerConfig()
    assert config.host == "127.0.0.1"
    assert config.api_key is None


@pytest.mark.unit
def test_warns_when_binding_non_loopback_without_api_key(captured_warnings: list[str]) -> None:
    _serve(OpenAIServerConfig(host="0.0.0.0", api_key=None))
    assert _has_insecure_warning(captured_warnings)


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        OpenAIServerConfig(host="127.0.0.1", api_key=None),
        OpenAIServerConfig(host="0.0.0.0", api_key="secret"),
    ],
)
def test_no_warning_on_loopback_or_with_api_key(captured_warnings: list[str], config: OpenAIServerConfig) -> None:
    _serve(config)
    assert not _has_insecure_warning(captured_warnings)
