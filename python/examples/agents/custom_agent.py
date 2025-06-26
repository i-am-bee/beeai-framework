import asyncio
import sys
import traceback

from pydantic import BaseModel, Field, InstanceOf

from beeai_framework.adapters.ollama import OllamaChatModel
from beeai_framework.agents import AgentMeta, BaseAgent, BaseAgentRunOptions
from beeai_framework.backend import AnyMessage, AssistantMessage, ChatModel, SystemMessage, UserMessage
from beeai_framework.context import Run, RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import BaseMemory, UnconstrainedMemory


class State(BaseModel):
    thought: str
    final_answer: str


class RunInput(BaseModel):
    message: InstanceOf[AnyMessage]


class CustomAgentRunOptions(BaseAgentRunOptions):
    max_retries: int | None = None


class CustomAgentRunOutput(BaseModel):
    message: InstanceOf[AnyMessage]
    state: State


class CustomAgent(BaseAgent[CustomAgentRunOutput]):
    def __init__(self, llm: ChatModel, memory: BaseMemory) -> None:
        super().__init__()
        self.model = llm
        self._memory = memory

    @property
    def memory(self) -> BaseMemory:
        return self._memory

    @memory.setter
    def memory(self, memory: BaseMemory) -> None:
        self._memory = memory

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["agent", "custom"],
            creator=self,
        )

    def run(
        self,
        run_input: RunInput,
        options: CustomAgentRunOptions | None = None,
    ) -> Run[CustomAgentRunOutput]:
        async def handler(context: RunContext) -> CustomAgentRunOutput:
            class CustomSchema(BaseModel):
                thought: str = Field(description="Describe your thought process before coming with a final answer")
                final_answer: str = Field(
                    description="Here you should provide concise answer to the original question."
                )

            response = await self.model.create_structure(
                schema=CustomSchema,
                messages=[
                    SystemMessage("You are a helpful assistant. Always use JSON format for your responses."),
                    *(self.memory.messages if self.memory is not None else []),
                    run_input.message,
                ],
                max_retries=options.max_retries if options else None,
                abort_signal=context.signal,
            )

            result = AssistantMessage(response.object["final_answer"])
            await self.memory.add(result) if self.memory else None

            return CustomAgentRunOutput(
                message=result,
                state=State(thought=response.object["thought"], final_answer=response.object["final_answer"]),
            )

        return self._to_run(
            handler, signal=options.signal if options else None, run_params={"input": run_input, "options": options}
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
        llm=OllamaChatModel("granite3.3:8b"),
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
