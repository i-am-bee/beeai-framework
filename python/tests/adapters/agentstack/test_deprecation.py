# Copyright 2026 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

# The warning is emitted once per process (on the first import of the package), so each scenario runs in a
# fresh interpreter to observe it regardless of what the test session has already imported.
SCENARIOS = {
    "direct import": "from beeai_framework.adapters.agentstack.agents import AgentStackAgent",
    "provider name": "from beeai_framework.backend import ChatModel\nChatModel.from_name('agentstack:model')",
    "beeai_platform alias": "from beeai_framework.adapters.beeai_platform.agents.agent import BeeAIPlatformAgent",
}


@pytest.mark.parametrize("code", SCENARIOS.values(), ids=SCENARIOS.keys())
def test_agentstack_import_warns_at_caller(code: str, tmp_path: Path) -> None:
    script = tmp_path / "user_code.py"
    script.write_text(code + "\n")

    result = subprocess.run(
        [sys.executable, "-W", "always::DeprecationWarning", str(script)],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    warnings = [line for line in result.stderr.splitlines() if "AgentStack is no longer maintained" in line]
    assert len(warnings) == 1, result.stderr
    # The warning must point at the user's code, not at framework internals.
    assert warnings[0].startswith(f"{script}:{len(code.splitlines())}: DeprecationWarning"), warnings[0]
