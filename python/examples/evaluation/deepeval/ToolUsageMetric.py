from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class ToolUsageMetric(BaseMetric):
    """
    Compares actual tool usage against expected tool usage.
    Scores 0.75 on tool existence match + 0.25 on usage count match.
    """

    success: bool = False

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.async_mode = False

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
            self.score = 1.0
            self.success = self.score >= self.threshold
            return self.score

        # Part 1: tool existence match (weight 0.75)
        matching_tools = sum(
            1 for tool in all_tools if tool in actual_tool_usage and tool in expected_tool_usage
        )
        existence_score = 0.75 * (matching_tools / len(all_tools))

        # Part 2: usage count match per expected tool (weight 0.25)
        count_scores = [
            1.0 if actual_tool_usage.get(tool, 0) == expected_tool_usage[tool] else 0.0
            for tool in expected_tool_usage
        ]
        count_score = 0.25 * (sum(count_scores) / len(count_scores)) if count_scores else 0.25

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
