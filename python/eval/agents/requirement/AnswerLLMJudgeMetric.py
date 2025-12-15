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


class AnswerLLMJudgeMetric(BaseMetric):
    """
    Uses an LLM as a judge to compare the actual answer vs the expected answer.
    Returns a semantic similarity score between 0 and 1.
    """

    success: bool = False  # ensure MetricData.success is always a bool

    def __init__(self, model: DeepEvalLLM | None = None, threshold: float = 0.5):
        super().__init__()
        # DeepEval expects model to be a DeepEvalBaseLLM; we use our wrapper.
        self.model: DeepEvalLLM = model or DeepEvalLLM.from_name("ollama:llama3.1:8b")
        self.threshold = threshold
        self.async_mode = True  # DeepEval will call a_measure

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ) -> float:
        actual = (test_case.actual_output or "").strip()
        expected = (test_case.expected_output or "").strip()

        # If no expected answer, treat as trivial pass/fail
        if not expected:
            score = 1.0 if not actual else 0.0
            self.score = score
            self.success = score >= self.threshold
            return score

        prompt = (
            "You are an evaluator.\n"
            "Compare the model's answer to the expected answer.\n\n"
            f"Question:\n{test_case.input}\n\n"
            f"Model answer:\n{actual}\n\n"
            f"Expected answer:\n{expected}\n\n"
            "Return ONLY a number between 0 and 1 (no text), where:\n"
            "0 = completely incorrect or unrelated,\n"
            "1 = fully correct and equivalent in meaning.\n"
        )

        text = await self.model.a_generate(prompt)  # DeepEvalLLM async call
        try:
            score = float(str(text).strip())
        except Exception:
            score = 0.0

        score = max(0.0, min(1.0, score))
        self.score = score
        self.success = score >= self.threshold
        return score

    def measure(self, test_case: LLMTestCase) -> float:
        """Sync wrapper in case something calls measure() directly."""
        import asyncio
        return asyncio.run(self.a_measure(test_case))

    def is_successful(self) -> bool:
        return getattr(self, "success", False)

    @property
    def __name__(self) -> str:
        return "AnswerLLMJudgeMetric"