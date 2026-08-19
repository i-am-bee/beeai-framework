# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.adapters.groq import GroqEmbeddingModel
from beeai_framework.backend.embedding import EmbeddingModel


@pytest.mark.unit
def test_groq_embedding_model_is_unsupported() -> None:
    """Groq has no embeddings API, so instantiation must fail with a clear error."""
    with pytest.raises(NotImplementedError, match="does not provide an embeddings API"):
        GroqEmbeddingModel()


@pytest.mark.unit
def test_groq_embedding_model_from_name_is_unsupported() -> None:
    """Resolving the embedding model by provider name surfaces the same clear error."""
    with pytest.raises(NotImplementedError, match="does not provide an embeddings API"):
        EmbeddingModel.from_name("groq")
