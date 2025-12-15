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

class FactsSimilarityMetric(BaseMetric):
    # Default so DeepEval's MetricData.success sees a proper boolean
    success: bool = False

    def __init__(self, model: DeepEvalLLM | None = None, threshold: float = 0.5):
        super().__init__()
        # DeepEval expects model to be a DeepEvalBaseLLM; we use our wrapper.
        self.model: DeepEvalLLM = model or DeepEvalLLM.from_name("ollama:llama3.1:8b")
        self.threshold = threshold
        # Let DeepEval use async execution path (a_measure)
        self.async_mode = True

    def _get_expected(self, test_case: LLMTestCase) -> list[str]:
        if hasattr(test_case, "expected_facts"):
            return getattr(test_case, "expected_facts")
        metadata = getattr(test_case, "additional_metadata", None) or {}
        return metadata.get("expected_facts", [])

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ) -> float:
        """Async entrypoint DeepEval actually uses; we ignore the extra flags."""
        actual_facts = getattr(test_case, "retrieval_context", [])
        expected_facts = self._get_expected(test_case)
        if not expected_facts:
            score = 1.0 if not actual_facts else 0.0
            self.score = score
            self.success = score >= self.threshold
            return score

        prompt = (
            "You are an evaluator.\n"
            "Compare the two lists of supporting facts.\n\n"
            f"Actual facts:\n{actual_facts}\n\n"
            f"Expected facts:\n{expected_facts}\n\n"
            "Return ONLY a number between 0 and 1 (no text), where:\n"
            "0 = completely different, 1 = identical in meaning.\n"
        )
        text = await self.model.a_generate(prompt)  # uses DeepEvalLLM.a_generate
        score = float(str(text).strip())
        score = max(0.0, min(1.0, score))
        self.score = score
        self.success = score >= self.threshold
        return score

    def measure(self, test_case: LLMTestCase) -> float:
        """Synchronous wrapper for environments that call measure() instead of a_measure()."""
        import asyncio

        return asyncio.run(self.a_measure(test_case))

    def is_successful(self) -> bool:
        return getattr(self, "success", False)

    @property
    def __name__(self):
        return "FactsSimilarityMetric"