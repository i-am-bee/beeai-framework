# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path

from pydantic import BaseModel, Field

from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AnyMessage, AssistantMessage, SystemMessage, UserMessage
from beeai_framework.workflows.v2.decorators._or import _or
from beeai_framework.workflows.v2.decorators.after import after
from beeai_framework.workflows.v2.decorators.end import end
from beeai_framework.workflows.v2.decorators.fork import fork
from beeai_framework.workflows.v2.decorators.join import join
from beeai_framework.workflows.v2.decorators.start import start
from beeai_framework.workflows.v2.decorators.when import when
from beeai_framework.workflows.v2.workflow import Workflow


class Task(BaseModel):
    task: str = Field("The task to perform.")
    context: str = Field("Context that is important when performing the task.")


class Step(BaseModel):
    """Stores information about a task."""

    id: str
    problem: str
    # dependencies: list["Step"]


class Plan(BaseModel):
    steps: list[Step] = Field(description="A list of steps necessary to complete the task.")


class StepResult(BaseModel):
    """Stores the result of a completed task."""

    id: str
    result: str


class Critique(BaseModel):
    """Stores critique feedback and score."""

    score: int = Field(description="Quality score between 0 and 100")
    suggestions: str


class ReWOOAgent(Workflow):
    def __init__(self) -> None:
        super().__init__()
        self.chat_model: ChatModel = ChatModel.from_name("ollama:ibm/granite4")
        self.task: Task | None = None
        self.current_plan: Plan | None = None
        self.results: list[StepResult] = []
        self.reset_plan_budget()

    def reset_plan_budget(self) -> None:
        self.replan_budget: int = 3

    @start
    async def process_input(self, input: list[AnyMessage]) -> None:
        """Extracts the task and context from the conversation."""
        print("Processing input")
        self.reset_plan_budget()
        result = await self.chat_model.run(
            [SystemMessage(content="Extract the task and any important context from the user message."), *input],
            response_format=Task,
        )
        self.task = Task(**result.output_structured.model_dump()) if result.output_structured else None

    @after(process_input)
    @when(lambda self: self.task is None)
    @end
    async def no_task(self) -> list[AnyMessage]:
        """Ends the workflow if there is no task to solve."""
        return [AssistantMessage("Hello, how can I help you?")]

    @after(process_input)
    @when(lambda self: self.task is not None)
    async def planner(self) -> list[Step]:
        """Creates a plan (list of steps) based on the task and context."""
        print("Planner")
        assert self.task is not None
        print(self.task.model_dump_json(indent=4))

        result = await self.chat_model.run(
            [
                SystemMessage(
                    "\n".join(
                        [
                            "Create a plan to address the given task.",
                            "The plan should incorporate the context and include a minimal number of steps.",
                            f"Task: {self.task.task}\n",
                            f"Context: {self.task.context}",
                        ]
                    )
                )
            ],
            response_format=Plan,
        )

        assert result.output_structured is not None
        self.plan = Plan(**result.output_structured.model_dump())
        return self.plan.steps

    @after(_or("planner", "replan"))
    @fork
    async def executor(self, step: Step) -> StepResult:
        """Executes plan steps in parallel"""
        print("Executor")
        prompt = f"Provide a short (2 line max) solution to the following task:\n Task: {step.problem}"
        result = await self.chat_model.run([UserMessage(prompt)])
        print("Executor complete")
        return StepResult(id=step.id, result=result.get_text_content())

    @after(executor)
    @join
    async def critique(self, steps: list[Step], results: list[StepResult]) -> Critique:
        print("Critique")
        self.results = results

        """Evaluates the plan and solutions."""
        assert self.task is not None
        assert self.plan is not None

        result = await self.chat_model.run(
            [
                AssistantMessage(
                    "\n".join(
                        [
                            "Provide a critical review of the following plan that is designed to solve the given task.",
                            "If the plan is not sufficient, propose concrete improvements.",
                            "The solution is only provided for context, do not critique it.",
                            "Score the plan between 0 and 100. Try to provide an accurate score.",
                            f"Task: {self.task.model_dump_json(indent=4)}",
                            f"Plan: {self.plan.model_dump_json(indent=4)}",
                            f"Solution: {[r.model_dump_json(indent=4) for r in results]}",
                        ]
                    )
                ),
            ],
            response_format=Critique,
        )

        assert result.output_structured is not None
        critique = Critique(**result.output_structured.model_dump())
        print(critique.model_dump_json(indent=4))
        return critique

    @after(critique)
    @when(lambda self, critique: critique.score < 90 and self.replan_budget > 0)
    async def replan(self, critique: Critique) -> list[Step]:
        """If the critique score is low, create a new plan based on feedback."""
        print("Replan")

        assert self.task is not None
        assert self.plan is not None

        result = await self.chat_model.run(
            [
                SystemMessage(
                    "\n".join(
                        [
                            "Revise the following plan based on the critique. You can edit existing steps, add steps or remove steps.",
                            f"Task: {self.task.model_dump_json(indent=4)}",
                            f"Current plan: {self.plan.model_dump_json(indent=4)}",
                            f"Critique: {critique.model_dump_json(indent=4)}",
                        ]
                    )
                )
            ],
            response_format=Plan,
        )
        self.replan_budget -= 1
        assert result.output_structured is not None
        self.plan = Plan(**result.output_structured.model_dump())
        return self.plan.steps

    @after(critique)
    @when(lambda self, critique: critique.score >= 90 or self.replan_budget == 0)
    async def solver(self) -> str:
        print("solver")

        assert self.plan is not None
        assert self.task is not None

        print(self.plan.model_dump_json(indent=4))

        solution = await self.chat_model.run(
            [
                SystemMessage(
                    f"Based on the plan and solutions, provide the final answer.\n"
                    f"Task: {self.task.model_dump_json(indent=4)}\n"
                    f"Plan: {self.plan.model_dump_json(indent=4)}\n"
                    f"Solution: {[r.model_dump_json(indent=4) for r in self.results]}"
                )
            ]
        )
        return solution.get_text_content()

    @end
    @after(solver)
    async def finalize(self, solution: str) -> list[AssistantMessage]:
        """Returns the final answer as a message."""
        print("Finalize")
        return [AssistantMessage(solution)]


# Async main function
async def main() -> None:
    workflow = ReWOOAgent()
    workflow.print_html(Path(__file__).resolve().parent / "workflow.html")
    output = await workflow.run([UserMessage("What is at the center of a black hole?")])
    print(output.last_message.text)


# Entry point
if __name__ == "__main__":
    asyncio.run(main())
