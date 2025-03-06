import asyncio
import sys
import traceback

from pydantic import BaseModel, Field, InstanceOf

from beeai_framework import (
    AssistantMessage,
    BaseAgent,
    BaseMemory,
    Message,
    SystemMessage,
    UnconstrainedMemory,
    UserMessage,
)
from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.agents.react.types import ReActAgentRunInput, ReActAgentRunOptions
from beeai_framework.agents.types import AgentMeta
from beeai_framework.backend.chat import ChatModel
from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.errors import FrameworkError
from beeai_framework.utils.models import ModelLike, to_model, to_model_optional


class State(BaseModel):
    thought: str
    final_answer: str


class RunOutput(BaseModel):
    message: InstanceOf[Message]
    state: State


class RunOptions(ReActAgentRunOptions):
    max_retries: int | None = None


class CustomAgent(BaseAgent[RunOutput]):
    memory: BaseMemory | None = None

    def __init__(self, llm: ChatModel, memory: BaseMemory) -> None:
        self.model = llm
        self.memory = memory

        self.emitter = Emitter.root().child(
            namespace=["agent", "custom"],
            creator=self,
        )

    async def _run(
        self,
        run_input: ModelLike[ReActAgentRunInput],
        options: ModelLike[ReActAgentRunOptions] | None,
        context: RunContext,
    ) -> RunOutput:
        run_input = to_model(ReActAgentRunInput, run_input)
        options = to_model_optional(ReActAgentRunOptions, options)

        class CustomSchema(BaseModel):
            thought: str = Field(description="Describe your thought process before coming with a final answer")
            final_answer: str = Field(description="Here you should provide concise answer to the original question.")

        response = await self.model.create_structure(
            schema=CustomSchema,
            messages=[
                SystemMessage("You are a helpful assistant. Always use JSON format for your responses."),
                *(self.memory.messages if self.memory is not None else []),
                UserMessage(run_input.prompt),
            ],
            max_retries=options.execution.total_max_retries if options and options.execution else None,
            abort_signal=context.signal,
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

    response = await agent.run("Why is the sky blue?")
    print(response.state)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
