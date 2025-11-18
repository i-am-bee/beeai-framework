# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, Field

from beeai_framework.backend.embedding import EmbeddingModel
from beeai_framework.backend.vector_store import VectorStore
from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools.search import SearchToolOutput, SearchToolResult
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.types import ToolRunOptions


class VectorStoreSearchToolInput(BaseModel):
    query: str = Field(description="Search query to find relevant documents in the vector store.")
    k: int = Field(description="Number of documents to retrieve, typically this argument is set to 5.", default=5)


class VectorStoreSearchToolResult(SearchToolResult):
    score: float = Field(description="Relevance score of the document.")


class VectorStoreSearchToolOutput(SearchToolOutput):
    def __init__(self, results: list[VectorStoreSearchToolResult]) -> None:
        super().__init__(results)  # type: ignore[arg-type]


class VectorStoreSearchTool(Tool[VectorStoreSearchToolInput, ToolRunOptions, VectorStoreSearchToolOutput]):
    name = "VectorStoreSearch"
    description = "Search for relevant documents in a vector store using semantic similarity."
    input_schema = VectorStoreSearchToolInput

    def __init__(self, vector_store: VectorStore, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)
        self.vector_store = vector_store

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "search", "retrieval", "vector_store"],
            creator=self,
        )

    async def _run(
        self, input: VectorStoreSearchToolInput, options: ToolRunOptions | None, context: RunContext
    ) -> VectorStoreSearchToolOutput:
        # Search the vector store using the provided query and parameters
        documents_with_score = await self.vector_store.search(query=input.query, k=input.k)

        # Convert DocumentWithScore objects to SearchToolResult format
        results = [
            VectorStoreSearchToolResult(
                title=str(doc.document.metadata.get("title", f"Document {i + 1}")),
                description=doc.document.content,
                url=str(doc.document.metadata.get("url", doc.document.metadata.get("source", ""))),
                score=doc.score,
            )
            for i, doc in enumerate(documents_with_score)
        ]

        return VectorStoreSearchToolOutput(results)

    @classmethod
    def from_vector_store_name(
        cls, name: str, *, embedding_model: EmbeddingModel, options: dict[str, Any] | None = None, **kwargs: Any
    ) -> VectorStoreSearchTool:
        """
        Create a VectorStoreSearchTool with a dynamically loaded vector store.

        Parameters
        ----------
        name : str
            A *case sensitive* string in the format "integration:ClassName".
            - `integration` is the name of the Python package namespace (e.g. "beeai").
            - `ClassName` is the name of the vector store class to load (e.g. "TemporalVectorStore").

        embedding_model : EmbeddingModel
            An instance of the embedding model required to initialize the vector store.

        options : dict[str, Any] | None
            Options to pass to the tool constructor.

        **kwargs :
            any positional or keywords arguments that would be passed to the vector store

        Returns
        -------
        VectorStoreSearchTool
            An instantiated search tool with the requested vector store.

        Raises
        ------
        ImportError
            If the specified vector store class cannot be found in any known integration package.
        """
        vector_store = VectorStore.from_name(name, embedding_model=embedding_model, **kwargs)
        return cls(vector_store=vector_store, options=options)

    async def clone(self) -> Self:
        cloned = await super().clone()
        cloned.vector_store = self.vector_store
        return cloned
