import sys
from pathlib import Path

from beeai_framework.tools.tool import Tool

from deepeval import evaluate
from deepeval.test_case import LLMTestCase

# Add the examples directory to sys.path to import setup_vector_store
examples_path = Path(__file__).parent.parent.parent.parent / "examples" / "agents" / "experimental" / "requirement"
sys.path.insert(0, str(examples_path))

#from examples.agents.experimental.requirement.rag import setup_vector_store
from beeai_framework.agents.experimental.agent import RequirementAgent
from deepeval.metrics import BaseMetric
from eval.model import DeepEvalLLM
from deepeval.test_case import LLMTestCase

class ToolUsageMetric(BaseMetric):
    """
    Compares actual tool usage against expected tool usage.
    expected_tool_usage בדוגמה:
        {"Wikipedia": 2, "PythonTool": 1}
    """

    # Default so DeepEval's MetricData.success sees a proper boolean
    success: bool = False

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.async_mode = False  # חובה כי אין measure אסינכרוני

    def _get_tool_usage(self, test_case: LLMTestCase, key: str) -> dict[str, int]:
        if hasattr(test_case, key):
            return getattr(test_case, key) or {}
        metadata = getattr(test_case, "additional_metadata", None) or {}
        return metadata.get(key, {}) or {}

    def measure(self, test_case: LLMTestCase) -> float:
        actual_tool_usage = self._get_tool_usage(test_case, "tool_usage")
        expected_tool_usage = self._get_tool_usage(test_case, "expected_tool_usage")

        all_tools = set(actual_tool_usage.keys()) | set(expected_tool_usage.keys())
        if not all_tools:
            score = 1.0
            self.score = score
            self.success = score >= self.threshold
            return score

        # =====================
        # חלק 1: השוואת כלים קיימים (0.75)
        # =====================
        matching_tools = sum(
            1 for tool in all_tools if tool in actual_tool_usage and tool in expected_tool_usage
        )
        existence_score = 0.75 * (matching_tools / len(all_tools))

        # =====================
        # חלק 2: השוואת כמויות שימוש בכל כלי (0.25)
        # =====================
        count_score_per_tool = []
        for tool in expected_tool_usage:
            actual_count = actual_tool_usage.get(tool, 0)
            expected_count = expected_tool_usage[tool]
            if actual_count == expected_count:
                count_score_per_tool.append(1.0)
            else:
                count_score_per_tool.append(0.0)
        if count_score_per_tool:
            count_score = 0.25 * (sum(count_score_per_tool) / len(count_score_per_tool))
        else:
            count_score = 0.25  # אין כלים צפוים → נותנים את הציון המלא לחלק זה

        # =====================
        # ציון סופי
        # =====================
        final_score = existence_score + count_score
        self.score = final_score
        self.success = final_score >= self.threshold
        return final_score


    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return getattr(self, "success", False)

    @property
    def __name__(self):
        return "ToolUsageMetric"