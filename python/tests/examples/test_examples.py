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
            "version.py",
            "helpers/io.py",
            # only test if API key is found
            "backend/providers/watsonx.py" if os.getenv("WATSONX_API_KEY") is None else None,
            "backend/providers/openai_example.py" if os.getenv("OPENAI_API_KEY") is None else None,
            # requires Searx instance
            "workflows/web_agent.py",
        ],
    )
)


def example_name(path: str) -> str:
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
