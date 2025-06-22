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

try:
    from langchain_core.documents import Document
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [langchain] not found.\nRun 'pip install \"beeai-framework[langchain]\"' to install."
    ) from e


from beeai_framework.adapters.langchain.wrappers.lc_embedding import LCEmbedding
from beeai_framework.backend.embedding import EmbeddingModel
from beeai_framework.backend.types import Document as VectorStoreDocument


def get_langchain_embedding(embedding_model: EmbeddingModel) -> LCEmbedding:
    return LCEmbedding(embedding=embedding_model)


def lc_document_to_document(lc_document: Document) -> VectorStoreDocument:
    return VectorStoreDocument(content=lc_document.page_content, metadata=lc_document.metadata)


def document_to_lc_document(document: VectorStoreDocument) -> Document:
    return Document(page_content=document.content, metadata=document.metadata)
