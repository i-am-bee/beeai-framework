
import asyncio
import json
import os
import sys
import pandas as pd
from typing import Any, List
from dataclasses import dataclass
from ragas.metrics.base import SingleTurnMetric, MetricType
from ragas.dataset_schema import SingleTurnSample
import typing as t

current_file_path = os.path.dirname(os.path.abspath(__file__)) # .../agents/requirement
eval_root = os.path.abspath(os.path.join(current_file_path, "..", "..")) 

if eval_root not in sys.path:
    sys.path.insert(0, eval_root)

from ragas.metrics.base import SingleTurnMetric





class ToolUsageMetric(SingleTurnMetric):
    name: str = "tool_usage"
    async def _ascore(self, row: dict, callbacks: Any = None) -> float:
        expected = row.get("expected_tools", [])
        actual = row.get("tools_called", [])
        if not expected: return 1.0 if not actual else 0.0
        matches = 0
        for exp in expected:
            exp_query = str(exp.get("input_parameters", {}).get("query", "")).lower()
            if any(exp_query in str(act.get("input_parameters", {}).get("query", "")).lower() for act in actual):
                matches += 1
        return float(matches / len(expected))

class FactsSimilarityMetric(SingleTurnMetric):
    name: str = "fact_similarity"
    def __init__(self, llm):
        self.llm = llm
        super().__init__()
    async def _ascore(self, row: dict, callbacks: Any = None) -> float:
        prompt = f"Reference: {row.get('reference')}\nResponse: {row.get('response')}\nScore 1.0 if facts match, 0.5 if partial, 0.0 if not. Return number only."
        try:
            res = await self.llm.generate_text(prompt)
            return float(''.join(c for c in res if c.isdigit() or c=='.'))
        except: return 0.0

