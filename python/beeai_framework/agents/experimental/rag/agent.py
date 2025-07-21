# Copyright 2025 © BeeAI a Series of LF Projects, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum

from pydantic import BaseModel, InstanceOf

from beeai_framework.agents import AgentExecutionConfig, AgentMeta, BaseAgent
from beeai_framework.backend import AnyMessage, AssistantMessage, ChatModel, SystemMessage, UserMessage
from beeai_framework.backend.types import DocumentWithScore
from beeai_framework.backend.vector_store import VectorStore
from beeai_framework.context import Run, RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.memory import BaseMemory
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from beeai_framework.retrieval.document_processors.document_processors import DocumentsRerankWithLLM
from beeai_framework.utils.cancellation import AbortSignal


class State(BaseModel):
    final_answer: str


class RagAgentRunInput(BaseModel):
    message: InstanceOf[AnyMessage]


class RAGAgentRunOutput(BaseModel):
    message: InstanceOf[AnyMessage]


class FallbackStrategy(str, Enum):
    NONE = "none"
    INTERNAL_KNOWLEDGE = "internal_knowledge"
    WEB_SEARCH = "web_search"


class RAGAgent(BaseAgent[RAGAgentRunOutput]):
    def __init__(
        self,
        *,
        llm: ChatModel,
        memory: BaseMemory,
        vector_store: VectorStore,
        reranker: DocumentsRerankWithLLM | None = None,
        number_of_retrieved_documents: int = 7,
        documents_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = llm
        self._memory = memory or UnconstrainedMemory()
        self.vector_store = vector_store
        self.reranker = reranker
        self.number_of_retrieved_documents = number_of_retrieved_documents
        self.documents_threshold = documents_threshold

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["agent", "rag"],
            creator=self,
        )

    def run(
        self,
        prompt: RagAgentRunInput,
        execution: AgentExecutionConfig | None = None,
        signal: AbortSignal | None = None,
    ) -> Run[RAGAgentRunOutput]:
        async def handler(context: RunContext) -> RAGAgentRunOutput:
            await self.memory.add(prompt.message)

            query = prompt.message.text
            retrieved_docs = await self.vector_store.search(query, k=self.number_of_retrieved_documents)

            # Apply re-ranking
            if self.reranker:
                retrieved_docs: list[DocumentWithScore] = await self.reranker.postprocess_documents(  # type: ignore[no-redef]
                    retrieved_docs, query=query
                )

            # Extract documents context
            docs_content = "\n\n".join(doc_with_score.document.content for doc_with_score in retrieved_docs)

            # Place content in template
            input_message = UserMessage(content=f"The context for replying to the query is:\n\n{docs_content}")

            messages = [
                SystemMessage("You are a helpful agent, answer based only on the context."),
                *self.memory.messages,
                input_message,
            ]
            response = await self.model.create(
                messages=messages,
                max_retries=execution.total_max_retries if execution else None,
                abort_signal=context.signal,
            )

            result = AssistantMessage(response.messages[-1].text)
            await self.memory.add(result)

            return RAGAgentRunOutput(message=result)

        return self._to_run(handler, signal=signal, run_params={"input": prompt, "execution": execution})

    @property
    def memory(self) -> BaseMemory:
        return self._memory

    @memory.setter
    def memory(self, memory: BaseMemory) -> None:
        self._memory = memory

    @property
    def meta(self) -> AgentMeta:
        return AgentMeta(
            name="RagAgent",
            description="Rag agent is an agent capable of answering questions based on a corpus of documents.",
            tools=[],
        )
