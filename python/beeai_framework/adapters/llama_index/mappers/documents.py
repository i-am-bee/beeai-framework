# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

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
    return DocumentWithScore(
        document=Document(content=document.text, metadata=document.metadata), score=document.score or 0.0
    )
