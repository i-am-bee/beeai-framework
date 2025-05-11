import asyncio
import os
from collections.abc import Callable
from pathlib import Path

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import LLMTestCase

from beeai_framework.agents.tool_calling import ToolCallingAgent
from eval.agents.tool_calling.utils import invoke_agent

ROOT_CACHE_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/.cache"
Path(ROOT_CACHE_DIR).mkdir(parents=True, exist_ok=True)


async def create_dataset(
    *, name: str, agent_factory: Callable[[], ToolCallingAgent], goldens: list[Golden], cache: bool
) -> EvaluationDataset:
    dataset = EvaluationDataset()
    cache_dir = Path(f"{ROOT_CACHE_DIR}/{name}")

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
            await invoke_agent(agent, case)
            return case

        for test_case in await asyncio.gather(*[process_golden(golden) for golden in goldens], return_exceptions=False):
            dataset.add_test_case(test_case)

        if cache:
            dataset.save_as(file_type="json", directory=str(cache_dir.absolute()), include_test_cases=True)

    for case in dataset.test_cases:
        case.name = f"{name} - {case.input[0:128].strip()}"

    return dataset


def create_dataset_sync(
    *, name: str, agent_factory: Callable[[], ToolCallingAgent], goldens: list[Golden], cache: bool | None = None
) -> EvaluationDataset:
    loop = asyncio.new_event_loop()
    dataset = loop.run_until_complete(
        create_dataset(
            name=name,
            agent_factory=agent_factory,
            goldens=goldens,
            cache=str(os.getenv("EVAL_CACHE_DATASET")).lower() == "true" if cache is None else cache,
        )
    )
    return dataset
