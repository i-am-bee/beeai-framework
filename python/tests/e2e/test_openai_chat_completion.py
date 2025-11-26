# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import contextlib
import threading
import time
from collections.abc import Generator
from typing import Unpack

import pytest
import requests

from beeai_framework.adapters.openai.serve._openai_model import OpenAIModel
from beeai_framework.adapters.openai.serve.server import OpenAIServer, OpenAIServerConfig, register
from beeai_framework.backend.message import AnyMessage, AssistantMessage
from beeai_framework.emitter import Emitter
from beeai_framework.runnable import Runnable, RunnableOptions, RunnableOutput, runnable_entry
from beeai_framework.serve.utils import UnlimitedMemoryManager


class DummyRunnable(Runnable[RunnableOutput]):
    @property
    def emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["dummy"])

    @runnable_entry
    async def run(self, input: list[AnyMessage], /, **kwargs: Unpack[RunnableOptions]) -> RunnableOutput:
        return RunnableOutput(output=[AssistantMessage(content="pong")])


class DummyModel(OpenAIModel):
    model_id = "dummy-model"

    def __init__(self) -> None:
        self.runnable_instance = DummyRunnable()
        super().__init__(runnable=self.runnable_instance, model_id=self.model_id)

    def get_runnable(self) -> DummyRunnable:
        return self.runnable_instance


@pytest.fixture(scope="session")
def start_test_server() -> Generator[str, None, None]:
    port = 19001

    register()
    with contextlib.suppress(Exception):
        OpenAIServer.register_factory(OpenAIModel, lambda model, **kwargs: model)  # type: ignore[type-var]

    config = OpenAIServerConfig(
        host="127.0.0.1",
        port=port,
        api_key="TEST_KEY",
    )

    server = OpenAIServer(config=config, memory_manager=UnlimitedMemoryManager())
    server.register(DummyModel())  # type: ignore[arg-type]

    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()

    server_url = f"http://127.0.0.1:{port}"

    for _ in range(20):  # Poll for up to 2 seconds
        try:
            requests.get(server_url, timeout=0.1)
            break  # Server is up
        except requests.exceptions.ConnectionError:
            time.sleep(0.1)
    else:
        pytest.fail(f"Server did not become reachable at {server_url}")

    yield server_url

    with contextlib.suppress(Exception):
        pass


@pytest.mark.e2e
def test_openai_e2e_success(start_test_server: str) -> None:
    url = f"{start_test_server}/chat/completions"

    payload = {
        "model": "dummy-model",
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
    }

    headers = {"Authorization": "Bearer TEST_KEY"}

    response = requests.post(url, json=payload, headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "pong"


@pytest.mark.e2e
def test_openai_e2e_auth_failure(start_test_server: str) -> None:
    url = f"{start_test_server}/chat/completions"

    payload = {
        "model": "dummy-model",
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
    }

    headers = {"Authorization": "Bearer WRONG"}

    response = requests.post(url, json=payload, headers=headers)

    assert response.status_code == 401
    assert "invalid" in response.text.lower() or "unauthorized" in response.text.lower()


@pytest.mark.e2e
def test_openai_e2e_model_not_found(start_test_server: str) -> None:
    url = f"{start_test_server}/chat/completions"

    payload = {
        "model": "non-existent-model",
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
    }

    headers = {"Authorization": "Bearer TEST_KEY"}

    response = requests.post(url, json=payload, headers=headers)

    assert response.status_code in [404, 400]
    assert "model" in response.text.lower()
