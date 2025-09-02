# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Unpack

from beeai_framework.agents import AgentMeta, AgentOptions, AgentOutput, BaseAgent
from beeai_framework.backend import AnyMessage, AssistantMessage, ChatModel, SystemMessage, UserMessage
from beeai_framework.backend.document_processor import DocumentProcessor
from beeai_framework.backend.types import DocumentWithScore
from beeai_framework.backend.vector_store import VectorStore
from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import BaseMemory
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from beeai_framework.runnable import runnable_entry


class RAGAgent(BaseAgent):
    def __init__(
        self,
        *,
        llm: ChatModel,
        memory: BaseMemory,
        vector_store: VectorStore,
        reranker: DocumentProcessor | None = None,
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

    @runnable_entry
    async def run(self, input: str | list[AnyMessage], /, **kwargs: Unpack[AgentOptions]) -> AgentOutput:
        """Execute the agent.

        Args:
            input: The input to the agent (if list of messages, uses the last message as input)
            total_max_retries: Maximum number of model retries.
            signal: The agent abort signal
            context: A dictionary that can be used to pass additional context to the agent

        Returns:
            The agent output.
        """
        if not input and self._memory.is_empty():
            raise ValueError(
                "Invalid input. The input must be a non-empty string or list of messages when memory is empty."
            )

        if isinstance(input, str):
            text_input = input
            await self.memory.add(UserMessage(text_input))
        else:
            await self.memory.add_many(input)
            text_input = input[-1].text if input else ""

        context = RunContext.get()

        try:
            retrieved_docs = await self.vector_store.search(text_input, k=self.number_of_retrieved_documents)

            # Apply re-ranking
            if self.reranker:
                retrieved_docs: list[DocumentWithScore] = await self.reranker.postprocess_documents(  # type: ignore[no-redef]
                    retrieved_docs, query=text_input
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
                max_retries=kwargs.get("total_max_retries"),
                abort_signal=context.signal,
            )

        except FrameworkError as error:
            error_message = AssistantMessage(content=error.explain())
            await self.memory.add(error_message)
            raise error

        result = response.messages[-1]
        await self.memory.add(result)
        return AgentOutput(output=[result])

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
