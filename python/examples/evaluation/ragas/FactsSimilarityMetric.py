"""Facts Similarity metric - Modern implementation with Ragas BaseMetric."""

import logging
import typing as t

import numpy as np

logger = logging.getLogger(__name__)
from pydantic import BaseModel, Field

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM


class FactsSimilarityOutput(BaseModel):
    """Output schema for facts similarity evaluation."""
    
    score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Similarity score between 0 and 1, where 0 = completely different, 1 = identical in meaning"
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of the similarity assessment"
    )


class FactsSimilarityMetric(BaseMetric):
    """
    Facts Similarity metric using LLM-based evaluation.

    Measures similarity between actual retrieved facts and expected reference facts.
    This metric uses an LLM judge to assess semantic similarity between two lists of facts.

    Rating scale: 0.0 (completely different) to 1.0 (identical in meaning)

    Usage:
        >>> from ragas.llms import llm_factory
        >>> from evaluation.adapters import InstructorRagasLLM
        >>> from FactsSimilarityMetric import FactsSimilarityMetric
        >>>
        >>> # Setup LLM
        >>> llm = InstructorRagasLLM.from_name("vertexai:gemini-2.0-flash-lite-001")
        >>>
        >>> # Create metric instance
        >>> metric = FactsSimilarityMetric(llm=llm)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     actual_facts=["Einstein was born in 1879", "He developed relativity theory"],
        ...     expected_facts=["Einstein was born in 1879", "Einstein created the theory of relativity"]
        ... )
        >>> print(f"Facts Similarity: {result.value}")

    Attributes:
        llm: Modern instructor-based LLM for evaluation
        name: The metric name
        allowed_values: Score range (0.0 to 1.0, higher is better)
        threshold: Minimum score to consider successful (default 0.5)
        max_retries: Maximum retry attempts for invalid ratings
    """

    # Type hints for linter (attributes are set in __init__)
    llm: "InstructorBaseRagasLLM"

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        name: str = "facts_similarity",
        threshold: float = 0.5,
        max_retries: int = 5,
        **kwargs,
    ):
        """
        Initialize FactsSimilarityMetric with required components.

        Args:
            llm: Modern instructor-based LLM for evaluation
            name: The metric name
            threshold: Minimum score to consider successful
            max_retries: Maximum retry attempts for invalid ratings
        """
        # Set attributes explicitly before calling super()
        self.llm = llm
        self.threshold = threshold
        self.max_retries = max_retries

        # Call super() for validation (without passing llm in kwargs)
        super().__init__(name=name, **kwargs)

    async def ascore(
        self, 
        actual_facts: t.List[str], 
        expected_facts: t.List[str]
    ) -> MetricResult:
        """
        Calculate facts similarity score using LLM-based evaluation.

        Args:
            actual_facts: The actual retrieved/generated facts to evaluate
            expected_facts: The expected reference facts (ground truth)

        Returns:
            MetricResult with facts similarity score (0.0-1.0, higher is better)
        """
        # Input validation with auto-conversion
        if not isinstance(actual_facts, list):
            if isinstance(actual_facts, str):
                actual_facts = [actual_facts]
            else:
                raise ValueError(f"actual_facts must be a list of strings, got {type(actual_facts)}")
        
        if not isinstance(expected_facts, list):
            if isinstance(expected_facts, str):
                expected_facts = [expected_facts]
            else:
                raise ValueError(f"expected_facts must be a list of strings, got {type(expected_facts)}")
        
        # Validate all items are strings
        actual_facts = [str(f) for f in actual_facts if f]  # Convert to strings and filter empty
        expected_facts = [str(f) for f in expected_facts if f]  # Convert to strings and filter empty

        # Handle empty cases
        if not expected_facts:
            score = 1.0 if not actual_facts else 0.0
            return MetricResult(value=float(score))

        if not actual_facts:
            return MetricResult(value=0.0)

        # Get similarity rating from LLM judge
        score = await self._get_similarity_rating(actual_facts, expected_facts)

        return MetricResult(value=float(score))

    async def _get_similarity_rating(
        self, 
        actual_facts: t.List[str], 
        expected_facts: t.List[str]
    ) -> float:
        """Get similarity rating from LLM judge with retry logic."""
        for retry in range(self.max_retries):
            try:
                # Format facts as numbered lists
                actual_str = "\n".join(f"{i+1}. {fact}" for i, fact in enumerate(actual_facts))
                expected_str = "\n".join(f"{i+1}. {fact}" for i, fact in enumerate(expected_facts))
                
                # Create evaluation prompt
                prompt = (
                    "You are an evaluator assessing the similarity between two lists of facts.\n\n"
                    "## Actual Facts:\n"
                    f"{actual_str}\n\n"
                    "## Expected Facts:\n"
                    f"{expected_str}\n\n"
                    "Compare the semantic meaning of both lists. Consider:\n"
                    "- Are the key facts present in both lists?\n"
                    "- Is the meaning preserved even if wording differs?\n"
                    "- Are there missing or extra facts?\n\n"
                    "Provide a similarity score where:\n"
                    "- 0.0 = Completely different facts\n"
                    "- 0.5 = Partially similar (some overlap)\n"
                    "- 1.0 = Identical meaning (even if worded differently)\n"
                )
                
                result = await self.llm.agenerate(prompt, FactsSimilarityOutput)
                score = result.score

                # Validate score is in expected range
                if 0.0 <= score <= 1.0:
                    return float(score)
                else:
                    # Invalid score - clamp and return
                    return float(max(0.0, min(1.0, score)))

            except Exception as e:
                if retry < self.max_retries - 1:
                    continue  # Retry on exception
                else:
                    logger.error("Facts Similarity evaluation failed after %d retries: %s", self.max_retries, e)
                    return float("nan")

        return float("nan")

    def is_successful(self, score: float) -> bool:
        """Check if the score meets the success threshold."""
        if np.isnan(score):
            return False
        return score >= self.threshold
