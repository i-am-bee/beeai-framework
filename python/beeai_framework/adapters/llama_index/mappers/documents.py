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
    from llama_index.core.schema import NodeWithScore, TextNode
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [llama_index] not found.\nRun 'pip install \"beeai-framework[llama_index]\"' to install."
    ) from e


from beeai_framework.backend.types import Document, DocumentWithScore


def doc_with_score_to_li_doc_with_score(document: DocumentWithScore) -> NodeWithScore:
    return NodeWithScore(
        node=TextNode(text=document.document.content, metadata=document.document.metadata), score=document.score
    )


def li_doc_with_score_to_doc_with_score(document: NodeWithScore) -> DocumentWithScore:
    return DocumentWithScore(document=Document(content=document.text, metadata=document.metadata), score=document.score)
