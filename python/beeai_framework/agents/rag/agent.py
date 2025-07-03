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

from beeai_framework.agents import AgentMeta, BaseAgent, BaseAgentRunOptions
from beeai_framework.agents.rag.prompts import QUERY_IMPROVEMENT_PROMPT
from beeai_framework.backend import AnyMessage, AssistantMessage, ChatModel, SystemMessage, UserMessage
from beeai_framework.backend.types import DocumentWithScore
from beeai_framework.backend.vector_store import VectorStore
from beeai_framework.context import Run, RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.memory import BaseMemory
from beeai_framework.retrieval.document_processors.document_processors import DocumentsRerankWithLLM


class State(BaseModel):
    final_answer: str


class RunInput(BaseModel):
    message: InstanceOf[AnyMessage]


class RAGAgentRunOptions(BaseAgentRunOptions):
    max_retries: int | None = None


class RAGAgentRunOutput(BaseModel):
    message: InstanceOf[AnyMessage]


class FallbackStrategy(str, Enum):
    NONE = "none"
    INTERNAL_KNOWLEDGE = "internal_knowledge"
    WEB_SEARCH = "web_search"


class RAGAgent(BaseAgent[RAGAgentRunOutput]):
    memory: BaseMemory | None = None

    def __init__(
        self,
        llm: ChatModel,
        memory: BaseMemory,
        vector_store: VectorStore,
        reranker: DocumentsRerankWithLLM | None = None,
    ) -> None:
        super().__init__()
        self.model = llm
        self.memory = memory
        self.vector_store = vector_store
        self.reranker = reranker

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
            await self.memory.add(run_input.message) if self.memory else None

            query = run_input.message.text
            retrieved_docs = await self.vector_store.search(query, k=10)

            # Apply re-ranking
            if self.reranker:
                retrieved_docs: list[DocumentWithScore] = await self.reranker.postprocess_documents(
                    query=query, documents=retrieved_docs
                )

            # Extract documents context
            docs_content = "\n\n".join(doc_with_score.document.content for doc_with_score in retrieved_docs)

            # Place content in template
            input_message = UserMessage(content=f"The context for replying to the query is:\n\n{docs_content}")

            messages = [
                SystemMessage("You are a helpful agent, answer based only on the context."),
                *(self.memory.messages if self.memory is not None else []),
                input_message,
            ]
            response = await self.model.create(
                messages=messages,
                max_retries=options.max_retries if options else None,
                abort_signal=context.signal,
            )

            result = AssistantMessage(response.messages[-1].text)
            await self.memory.add(result) if self.memory else None

            return RAGAgentRunOutput(message=result)

        return self._to_run(
            handler, signal=options.signal if options else None, run_params={"input": run_input, "options": options}
        )

    async def _search_with_scores(self, query: str) -> list[DocumentWithScore]:
        next_search_technique = "rerank"
        search_round = 0
        while search_round < 5:
            search_round += 1
            match next_search_technique:
                case "rerank":
                    retrieved_docs = await self.vector_store.search(query, k=self.number_of_retrieved_documents)
                    next_search_technique = "more_documents"
                case "more_documents":
                    retrieved_docs = await self.vector_store.search(
                        query, k=self.number_of_retrieved_documents * search_round
                    )
                    next_search_technique = "rephrase_search"
                case "rephrase_search":
                    rephrasing_prompt = [UserMessage(content=QUERY_IMPROVEMENT_PROMPT.render(user_query=query))]
                    search_query = await self.model.create(messages=rephrasing_prompt)
                    search_query = search_query.messages[-1].text
                    print(search_query)
                    retrieved_docs = await self.vector_store.search(search_query, k=self.number_of_retrieved_documents)
                    next_search_technique = "more_documents"

            retrieved_docs: list[DocumentWithScore] = await self.reranker.postprocess_documents(
                query=query, documents=retrieved_docs
            )
            retrieved_docs = [doc for doc in retrieved_docs if doc.score >= self.documents_threshold]
            if retrieved_docs:
                return retrieved_docs
        return []

    @property
    def meta(self) -> AgentMeta:
        return AgentMeta(
            name="RagAgent",
            description="Rag agent is an agent capable of answering questions based on a corpus of documents.",
            tools=[],
        )
