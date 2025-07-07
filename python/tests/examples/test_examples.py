import os
import pathlib
import runpy

import pytest
from dotenv import load_dotenv

load_dotenv()

EXAMPLES_DIR = (pathlib.Path(__file__).parent.parent.parent / "examples").resolve()
all_examples = list(EXAMPLES_DIR.rglob("*.py"))

exclude = list(
    filter(
        None,
        [
            "_*.py",
            "helpers/io.py",
            "backend/providers/watsonx.py" if os.getenv("WATSONX_API_KEY") is None else None,
            "backend/providers/mistralai.py" if os.getenv("MISTRALAI_API_KEY") is None else None,
            "backend/providers/ollama.py" if os.getenv("OLLAMA_BASE_URL") is None else None,
            "backend/embedding.py" if os.getenv("OLLAMA_BASE_URL") is None else None,
            "backend/providers/openai_example.py" if os.getenv("OPENAI_API_KEY") is None else None,
            "backend/providers/groq.py" if os.getenv("GROQ_API_KEY") is None else None,
            "backend/providers/xai.py" if os.getenv("XAI_API_KEY") is None else None,
            "backend/providers/vertexai.py" if os.getenv("GOOGLE_VERTEX_PROJECT") is None else None,
            "backend/providers/amazon_bedrock.py" if os.getenv("AWS_ACCESS_KEY_ID") is None else None,
            "backend/providers/anthropic.py" if os.getenv("ANTHROPIC_API_KEY") is None else None,
            "backend/providers/azure_openai.py" if os.getenv("AZURE_API_KEY") is None else None,
            "tools/mcp_agent.py" if os.getenv("SLACK_BOT_TOKEN") is None else None,
            "tools/mcp_tool_creation.py" if os.getenv("SLACK_BOT_TOKEN") is None else None,
            "tools/mcp_slack_agent.py" if os.getenv("SLACK_BOT_TOKEN") is None else None,
            "workflows/searx_agent.py",
            "agents/providers/acp.py",
            "agents/providers/a2a_agent.py",
            "agents/providers/beeai_platform.py",
            "agents/providers/watsonx_orchestrate.py",
            "workflows/remote.py",
            "serve/acp.py",
            "serve/beeai_platform.py",
            "serve/beeai_platform_await.py",
            "serve/a2a_server.py",
            "serve/acp_with_custom_agent.py",
            "serve/mcp_tool.py",
            "serve/watsonx_orchestrate.py",
            "tools/python_tool.py" if os.getenv("CODE_INTERPRETER_URL") is None else None,
            "tools/custom/sandbox.py" if os.getenv("CODE_INTERPRETER_URL") is None else None,
            "workflows/travel_advisor.py",
            "playground/*.py",
            "playground/*/*.py",
            "playground/*/*/*.py",
            "agents/experimental/requirement/exercises/*",
            "integrations/langgraph_example.py" if os.getenv("OLLAMA_API_BASE") else None,
            # Interactive example
            "agents/experimental/requirement/multi_agent.py",
        ],
    )
)


def example_name(path: pathlib.Path) -> str:
    return str(path.relative_to(EXAMPLES_DIR)).replace(os.sep, "/")


def is_excluded(path: pathlib.Path) -> bool:
    for pattern in exclude:
        if "/**" in pattern:
            raise ValueError("Double star '**' is not supported!")

        if path.match(pattern):
            return True
    return False


examples = sorted(
    {example for example in all_examples if not is_excluded(example)},
    key=example_name,
)


@pytest.mark.e2e
def test_finds_examples() -> None:
    assert examples


@pytest.mark.e2e
@pytest.mark.parametrize("example", examples, ids=example_name)
def test_example_execution(example: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = iter(["Hello world", "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    runpy.run_path(str(example.resolve()), run_name="__main__")
