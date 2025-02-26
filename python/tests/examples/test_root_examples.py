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

import pytest
from pytest_console_scripts import ScriptRunner


@pytest.mark.e2e
def test_basic(script_runner: ScriptRunner) -> None:
    result = script_runner.run(["examples/basic.py"])
    assert result.returncode == 0
    assert "Boston" in result.stdout


@pytest.mark.e2e
def test_llms(script_runner: ScriptRunner) -> None:
    result = script_runner.run(["examples/llms.py", "ollama"])
    assert result.returncode == 0
    assert "Brava" in result.stdout
