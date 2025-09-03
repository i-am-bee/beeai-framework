import asyncio
import sys
import traceback
from typing import Unpack

from pydantic import BaseModel, Field

from beeai_framework.adapters.ollama import OllamaChatModel
from beeai_framework.agents import AgentMeta, AgentOptions, AgentOutput, BaseAgent
from beeai_framework.backend import AnyMessage, AssistantMessage, ChatModel, SystemMessage, UserMessage
from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import BaseMemory, UnconstrainedMemory
from beeai_framework.runnable import runnable_entry


class State(BaseModel):
    thought: str
    final_answer: str


class CustomAgent(BaseAgent):
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

    @runnable_entry
    async def run(self, input: str | list[AnyMessage], /, **kwargs: Unpack[AgentOptions]) -> AgentOutput:
        async def handler(context: RunContext) -> AgentOutput:
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
                    *([UserMessage(input)] if isinstance(input, str) else input),
                ],
                max_retries=kwargs.get("total_max_retries", 3),
                abort_signal=context.signal,
            )

            result = AssistantMessage(response.object["final_answer"])
            await self.memory.add(result) if self.memory else None

            return AgentOutput(
                output=[result],
                context={
                    "state": State(thought=response.object["thought"], final_answer=response.object["final_answer"])
                },
            )

        return await handler(RunContext.get())

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

    response = await agent.run([UserMessage("Why is the sky blue?")])
    print(response.context.get("state"))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
