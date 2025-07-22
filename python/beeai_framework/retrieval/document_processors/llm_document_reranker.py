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


from beeai_framework.adapters.llama_index.mappers.chat import LlamaIndexChatModel
from beeai_framework.adapters.llama_index.mappers.documents import (
    doc_with_score_to_li_doc_with_score,
    li_doc_with_score_to_doc_with_score,
)
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.documentprocessor import DocumentProcessor
from beeai_framework.backend.types import DocumentWithScore

try:
    from llama_index.core.postprocessor.llm_rerank import LLMRerank
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [llama_index] not found.\nRun 'pip install \"beeai-framework[llama_index]\"' to install."
    ) from e


class LLMDocumentReranker(DocumentProcessor):
    llm: ChatModel

    def __init__(self, llm: ChatModel, *, choice_batch_size: int = 5, top_n: int = 5) -> None:
        self.llm = llm
        self.reranker = LLMRerank(
            choice_batch_size=choice_batch_size, top_n=top_n, llm=LlamaIndexChatModel(llm=self.llm)
        )

    async def postprocess_documents(
        self, documents: list[DocumentWithScore], *, query: str | None = None
    ) -> list[DocumentWithScore]:
        if query is None:
            raise ValueError("DocumentsRerankWithLLM requires 'query' parameter for reranking")

        li_documents_with_score = [doc_with_score_to_li_doc_with_score(document) for document in documents]
        processed_nodes = await self.reranker.apostprocess_nodes(li_documents_with_score, query_str=query)
        documents_with_score = [li_doc_with_score_to_doc_with_score(node) for node in processed_nodes]
        return documents_with_score
