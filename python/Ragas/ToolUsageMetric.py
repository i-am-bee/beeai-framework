from typing import Any, List, Optional

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult


class ToolUsageMetric(BaseMetric):
    """
    A metric to evaluate the correct usage of tools by an LLM in its responses.
    
    Compares expected tool calls with actual tool calls based on:
    - Tool name matching
    - Query parameter substring matching
    
    Args:
        name: The name of the metric
        threshold: Minimum score (0.0-1.0) required for success
    """
    def __init__(
        self,
        name: str = "tool_usage_metric",
        threshold: float = 0.5,
        **base_kwargs,
    ):
        """Initialize ToolUsageMetric."""
        super().__init__(name=name, **base_kwargs)
        self.threshold = threshold
        self.score = 0.0
        self.success = False
    
    def measure(self, response: Any, reference: Any) -> float:
        """Measure tool usage accuracy between expected and actual tools."""
        # Extract expected tools from response object
        expected_tools = getattr(response, "expected_tools", []) or \
                         (response.additional_metadata.get("expected_tools_detail") 
                          if hasattr(response, "additional_metadata") and response.additional_metadata else [])
        
        # Extract actual tools from response object
        actual_tools = getattr(reference, "tools_called", []) or \
                       (reference.additional_metadata.get("actual_tools_detail") 
                        if hasattr(reference, "additional_metadata") and reference.additional_metadata else [])
        
        if not expected_tools:
            self.score = 1.0 if not actual_tools else 0.0
            self.success = self.score >= self.threshold
            return self.score

        matches = 0
        used_actual_indices = set()

        for expected in expected_tools:
            exp_name = getattr(expected, "name", "")
            exp_params = getattr(expected, "input_parameters", {}) or {}
            exp_query = str(exp_params.get("query", "")).lower()
            
            for i, actual in enumerate(actual_tools):
                if i in used_actual_indices:
                    continue
                
                act_name = getattr(actual, "name", "")
                act_params = getattr(actual, "input_parameters", {}) or {}
                act_query = str(act_params.get("query", "")).lower()
                
                if exp_name == act_name and exp_query in act_query:
                    matches += 1
                    used_actual_indices.add(i)
                    break
        
        self.score = matches / len(expected_tools)
        self.success = self.score >= self.threshold
        return self.score
    
    async def ascore(
        self,
        reference: Any,
        response: Any,
        tools_used: Optional[List] = None
    ) -> MetricResult:
        """Async version of measure that returns MetricResult."""
        score = self.measure(response, reference)
        return MetricResult(value=score)

            
    
    
    