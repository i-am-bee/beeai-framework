import asyncio
import sys
import traceback

from pydantic import BaseModel, Field, InstanceOf

from beeai_framework import (
    AssistantMessage,
    BaseAgent,
    BaseMemory,
    SystemMessage,
    UnconstrainedMemory,
    UserMessage,
)
from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.agents.base import run_context
from beeai_framework.agents.types import AgentMeta
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AnyMessage
from beeai_framework.emitter import Emitter
from beeai_framework.errors import FrameworkError


class State(BaseModel):
    thought: str
    final_answer: str


class RunInput(BaseModel):
    message: InstanceOf[AnyMessage]


class RunOptions(BaseModel):
    max_retries: int | None = None


class RunOutput(BaseModel):
    message: InstanceOf[AnyMessage]
    state: State


class CustomAgent(BaseAgent[RunInput, RunOptions, RunOutput]):
    memory: BaseMemory | None = None

    def __init__(self, llm: ChatModel, memory: BaseMemory) -> None:
        self.model = llm
        self.memory = memory

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["agent", "custom"],
            creator=self,
        )

    @run_context
    async def run(
        self,
        run_input: RunInput,
        options: RunOptions | None = None,
    ) -> RunOutput:
        class CustomSchema(BaseModel):
            thought: str = Field(description="Describe your thought process before coming with a final answer")
            final_answer: str = Field(description="Here you should provide concise answer to the original question.")

        response = await self.model.create_structure(
            schema=CustomSchema,
            messages=[
                SystemMessage("You are a helpful assistant. Always use JSON format for your responses."),
                *(self.memory.messages if self.memory is not None else []),
                run_input.message,
            ],
            max_retries=options.max_retries if options else None,
            abort_signal=self._run_context.signal if self._run_context else None,
        )

        result = AssistantMessage(response.object["final_answer"])
        await self.memory.add(result) if self.memory else None

        return RunOutput(
            message=result,
            state=State(thought=response.object["thought"], final_answer=response.object["final_answer"]),
        )

    @property
    def meta(self) -> AgentMeta:
        return AgentMeta(
            name="CustomAgent",
            description="Custom Agent is a simple LLM agent.",
            tools=[],
        )


async def main() -> None:
    agent = CustomAgent(
        llm=OllamaChatModel("granite3.1-dense:8b"),
        memory=UnconstrainedMemory(),
    )

    response = await agent.run(RunInput(message=UserMessage("Why is the sky blue?")))
    print(response.state)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
