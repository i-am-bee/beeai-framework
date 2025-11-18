# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TypeVar

import pytest
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate import DisplayConfig
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.test_run.test_run import TestRunResultDisplay
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table

from beeai_framework.agents import AnyAgent

ROOT_CACHE_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/.cache"
Path(ROOT_CACHE_DIR).mkdir(parents=True, exist_ok=True)


T = TypeVar("T", bound=AnyAgent)


async def create_dataset(
    *,
    name: str,
    agent_factory: Callable[[], T],
    agent_run: Callable[[T, LLMTestCase], Awaitable[None]],
    goldens: list[Golden],
    cache: bool | None = None,
) -> EvaluationDataset:
    dataset = EvaluationDataset()

    cache_dir = Path(f"{ROOT_CACHE_DIR}/{name}")
    if cache is None:
        cache = os.getenv("EVAL_CACHE_DATASET", "").lower() == "true"

    if cache and cache_dir.exists():
        for file_path in cache_dir.glob("*.json"):
            dataset.add_test_cases_from_json_file(
                file_path=str(file_path.absolute().resolve()),
                input_key_name="input",
                actual_output_key_name="actual_output",
                expected_output_key_name="expected_output",
                context_key_name="context",
                tools_called_key_name="tools_called",
                expected_tools_key_name="expected_tools",
                retrieval_context_key_name="retrieval_context",
            )
    else:

        async def process_golden(golden: Golden) -> LLMTestCase:
            agent = agent_factory()
            case = LLMTestCase(
                input=golden.input,
                expected_tools=golden.expected_tools,
                actual_output="",
                expected_output=golden.expected_output,
                comments=golden.comments,
                context=golden.context,
                tools_called=golden.tools_called,
                retrieval_context=golden.retrieval_context,
                additional_metadata=golden.additional_metadata,
            )
            await agent_run(agent, case)
            return case

        for test_case in await asyncio.gather(*[process_golden(golden) for golden in goldens], return_exceptions=False):
            dataset.add_test_case(test_case)

        if cache:
            dataset.save_as(file_type="json", directory=str(cache_dir.absolute()), include_test_cases=True)

    for case in dataset.test_cases:
        case.name = f"{name} - {case.input[0:128].strip()}"  # type: ignore

    return dataset


def evaluate_dataset(
    dataset: EvaluationDataset, metrics: list[BaseMetric], display_mode: TestRunResultDisplay | None = None
) -> None:
    console = Console()
    console.print("[bold green]Evaluating dataset[/bold green]")

    if display_mode is None:
        display_mode = TestRunResultDisplay(os.environ.get("EVAL_DISPLAY_MODE", "all"))

    output = evaluate(
        test_cases=dataset.test_cases,  # type: ignore
        metrics=metrics,
        display_config=DisplayConfig(
            show_indicator=False, print_results=False, verbose_mode=False, display_option=None
        ),
    )

    # Calculate pass/fail counts
    total = len(output.test_results)
    passed = sum(
        bool(test_result.metrics_data) and all(md.success for md in (test_result.metrics_data or []))
        for test_result in output.test_results
    )
    failed = total - passed

    # Print summary table
    summary_table = Table(title="Test Results Summary", show_header=True, header_style="bold cyan")
    summary_table.add_column("Total", justify="right")
    summary_table.add_column("Passed", justify="right", style="green")
    summary_table.add_column("Failed", justify="right", style="red")
    summary_table.add_row(str(total), str(passed), str(failed))
    console.print(summary_table)

    for test_result in output.test_results:
        if display_mode != TestRunResultDisplay.ALL and (
            (display_mode == TestRunResultDisplay.FAILING and test_result.success)
            or (display_mode == TestRunResultDisplay.PASSING and not test_result.success)
        ):
            continue

        # Info Table
        info_table = Table(show_header=False, box=None, pad_edge=False)
        info_table.add_row("Input", str(test_result.input))
        info_table.add_row("Expected Output", str(test_result.expected_output))
        info_table.add_row("Actual Output", str(test_result.actual_output))

        # Metrics Table
        metrics_table = Table(title="Metrics", show_header=True, header_style="bold magenta")
        metrics_table.add_column("Metric")
        metrics_table.add_column("Success")
        metrics_table.add_column("Score")
        metrics_table.add_column("Threshold")
        metrics_table.add_column("Reason")
        metrics_table.add_column("Error")
        # metrics_table.add_column("Verbose Log")

        for metric_data in test_result.metrics_data or []:
            metrics_table.add_row(
                str(metric_data.name),
                str(metric_data.success),
                str(metric_data.score),
                str(metric_data.threshold),
                str(metric_data.reason),
                str(metric_data.error) if metric_data.error else "",
                # str(metric_data.verbose_logs),
            )

        # Print the panel with info and metrics table
        console.print(
            Panel(
                Group(info_table, metrics_table),
                title=f"[bold blue]{test_result.name}[/bold blue]",
                border_style="blue",
            )
        )

    # Gather failed tests
    if failed:
        pytest.fail(f"{failed}/{total} tests failed. See the summary table above for more details.", pytrace=False)
    else:
        assert 1 == 1
