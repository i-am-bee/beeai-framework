import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.agent import create_agent as _create_agent
from beeai_framework.agents.requirement import RequirementAgent


def create_agent() -> RequirementAgent:
    """Create agent using VertexAI (Ragas pipeline default)."""
    return _create_agent(model_name="vertexai:gemini-2.5-flash")
