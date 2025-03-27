# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pathlib
import runpy

import pytest
from dotenv import load_dotenv

load_dotenv()

all_examples = list(pathlib.Path(__file__, "../../../examples").resolve().rglob("*.py"))

exclude = list(
    filter(
        None,
        [
            # Dont test helper code
            "helpers/io.py",
            # Only test authenticated providers if API key is found
            "backend/providers/watsonx.py" if os.getenv("WATSONX_API_KEY") is None else None,
            "backend/providers/openai_example.py" if os.getenv("OPENAI_API_KEY") is None else None,
            "backend/providers/groq.py" if os.getenv("GROQ_API_KEY") is None else None,
            "backend/providers/xai.py" if os.getenv("XAI_API_KEY") is None else None,
            # Google backend picks up environment variables/google auth credentials directly
            "backend/providers/vertexai.py" if os.getenv("GOOGLE_VERTEX_PROJECT") is None else None,
            "backend/providers/amazon_bedrock.py" if os.getenv("AWS_ACCESS_KEY_ID") is None else None,
            "backend/providers/anthropic.py" if os.getenv("ANTHROPIC_API_KEY") is None else None,
            "backend/providers/azure_openai.py" if os.getenv("AZURE_API_KEY") is None else None,
            # MCP examples require Slack bot
            "tools/mcp_agent.py" if os.getenv("SLACK_BOT_TOKEN") is None else None,
            "tools/mcp_tool_creation.py" if os.getenv("SLACK_BOT_TOKEN") is None else None,
            "tools/mcp_slack_agent.py" if os.getenv("SLACK_BOT_TOKEN") is None else None,
            # Example requires Searx instance
            "workflows/searx_agent.py",
            # Requires BeeAI platform to be running
            "agents/experimental/remote.py",
            "workflows/remote.py",
            # Requires Code Interpreter to be running
            "tools/python_tool.py" if os.getenv("CODE_INTERPRETER_URL") is None else None,
            "tools/custom/sandbox.py" if os.getenv("CODE_INTERPRETER_URL") is None else None,
        ],
    )
)


def example_name(path: pathlib.Path) -> str:
    return os.path.relpath(path, start="examples")


examples = sorted({example for example in all_examples if example_name(example) not in exclude}, key=example_name)


@pytest.mark.e2e
def test_finds_examples() -> None:
    assert examples


@pytest.mark.e2e
@pytest.mark.parametrize("example", examples, ids=example_name)
def test_example_execution(example: str, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = iter(["Hello world", "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    runpy.run_path(example, run_name="__main__")
