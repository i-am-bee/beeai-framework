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


from beeai_framework.adapters.llama_index.utils import doc_with_score_to_li_doc_with_score, li_doc_with_score_to_doc_with_score
from beeai_framework.adapters.llama_index.wrappers.li_llm import LILLM
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.document_processor import DocumentProcessor
from beeai_framework.backend.types import DocumentWithScore

try:
    from llama_index.core.postprocessor.llm_rerank import LLMRerank
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [llama_index] not found.\nRun 'pip install \"beeai-framework[llama_index]\"' to install."
    ) from e


class DocumentsRerankWithLLM(DocumentProcessor):
    llm: ChatModel
    
    def __init__(self, llm: ChatModel):
        self.llm = llm
        self.reranker = LLMRerank(choice_batch_size=5, top_n=5, llm=LILLM(bai_llm=self.llm))
        
    def postprocess_documents(self, query: str, documents: list[DocumentWithScore]) -> list[DocumentWithScore]:
        li_documents_with_score = [doc_with_score_to_li_doc_with_score(document) for document in documents]
        processed_nodes = self.reranker.postprocess_nodes(
            li_documents_with_score, query_str=query
        )
        documents_with_score = [li_doc_with_score_to_doc_with_score(node) for node in processed_nodes]
        return documents_with_score
    
    async def apostprocess_nodes(self, query: str, documents: list[DocumentWithScore]) -> list[DocumentWithScore]:
        li_documents_with_score = [doc_with_score_to_li_doc_with_score(document) for document in documents]
        processed_nodes = await self.reranker.apostprocess_nodes(
            li_documents_with_score, query_str=query
        )
        documents_with_score = [li_doc_with_score_to_doc_with_score(node) for node in processed_nodes]
        return documents_with_score
    