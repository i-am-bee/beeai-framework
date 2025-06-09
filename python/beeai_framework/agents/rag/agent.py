
from beeai_framework.backend.vector_store import VectorStore
from pydantic import BaseModel, Field, InstanceOf
from beeai_framework.agents import AgentMeta, BaseAgent, BaseAgentRunOptions
from beeai_framework.backend import AnyMessage, AssistantMessage, ChatModel, SystemMessage, UserMessage
from beeai_framework.context import Run, RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.memory import BaseMemory


class State(BaseModel):
    thought: str
    final_answer: str


class RunInput(BaseModel):
    message: InstanceOf[AnyMessage]


class RAGAgentRunOptions(BaseAgentRunOptions):
    max_retries: int | None = None


class RAGAgentRunOutput(BaseModel):
    message: InstanceOf[AnyMessage]
    state: State

class RAGAgent(BaseAgent[RAGAgentRunOutput]):
    memory: BaseMemory | None = None

    def __init__(self, llm: ChatModel, memory: BaseMemory, vector_store: VectorStore) -> None:
        super().__init__()
        self.model = llm
        self.memory = memory
        self.vector_store = vector_store
        self.emitter.on("*.*", lambda data, event: print("Got something: ", data, event))

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["agent", "rag"],
            creator=self,
        )

    def run(
        self,
        run_input: RunInput,
        options: RAGAgentRunOptions | None = None,
    ) -> Run[RAGAgentRunOutput]:
        async def handler(context: RunContext) -> RAGAgentRunOutput:
            # class CustomSchema(BaseModel):
            #     thought: str = Field(description="Describe your thought process before coming with a final answer")
            #     final_answer: str = Field(
            #         description="Here you should provide concise answer to the original question."
            #     )
            
            await self.memory.add(run_input.message) if self.memory else None
            retrieved_docs = await self.vector_store.asearch(run_input.message.text)
            # Extract documents context
            docs_content = "\n\n".join(doc_with_score[0].content for doc_with_score in retrieved_docs)
        
            # Place content in template
            input_message = UserMessage(content=f"The context for replying to the task is:\n\n{docs_content}")
            
            messages=[
                    SystemMessage("You are a helpful agent, answer based only on the context."),
                    *(self.memory.messages if self.memory is not None else []),
                    input_message,
                ]
            response = await self.model.create(
            # response = await self.model.create_structure(
                # schema=CustomSchema,
                messages=messages,
                max_retries=options.max_retries if options else None,
                abort_signal=context.signal,
            )

            # result = AssistantMessage(response.object["final_answer"])
            result = AssistantMessage(response.messages[-1].text)
            await self.memory.add(result) if self.memory else None

            return RAGAgentRunOutput(
                message=result,
                # state=State(thought=response.object["thought"], final_answer=response.object["final_answer"]),
                state=State(thought="The joke!", final_answer=result.text),
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


