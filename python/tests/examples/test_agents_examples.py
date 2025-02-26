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

import io

import pytest
from pytest import MonkeyPatch
from pytest_console_scripts import ScriptRunner


@pytest.mark.e2e
def test_bee(script_runner: ScriptRunner, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("Hello"))
    result = script_runner.run(["examples/agents/bee.py"])
    assert result.returncode == 0


# TODO: figure out why this one doesn't work through pytest
@pytest.mark.skip(reason="TODO: something funky going on between the test and examples/helpers/io")
@pytest.mark.e2e
def test_granite(script_runner: ScriptRunner, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("Hello"))
    result = script_runner.run(["examples/agents/granite.py"])
    assert result.returncode == 0


@pytest.mark.e2e
def test_simple(script_runner: ScriptRunner) -> None:
    result = script_runner.run(["examples/agents/simple.py"])
    assert result.returncode == 0
    assert "Las Vegas" in result.stdout
