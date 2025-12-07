# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.embedding import EmbeddingModel


class Backend:
    """A unified interface for managing chat and embedding models.

    The Backend class provides a convenient way to bundle together a chat model
    and an embedding model, which are commonly used together in AI applications.
    It offers factory methods for easy instantiation from provider names.

    Attributes:
        chat: The chat model instance for text generation and conversation.
        embedding: The embedding model instance for text vectorization.

    Example:
        >>> # Create backend with specific models
        >>> from beeai_framework.backend import Backend
        >>> from beeai_framework.adapters.openai import OpenAIChatModel, OpenAIEmbeddingModel
        >>> backend = Backend(
        ...     chat=OpenAIChatModel("gpt-4"),
        ...     embedding=OpenAIEmbeddingModel("text-embedding-3-small")
        ... )
        
        >>> # Create backend from provider names
        >>> backend = Backend.from_name(chat="openai:gpt-4", embedding="openai:text-embedding-3-small")
        
        >>> # Create backend using same provider for both
        >>> backend = Backend.from_provider("openai")
    """

    def __init__(self, *, chat: ChatModel, embedding: EmbeddingModel) -> None:
        """Initialize a Backend with chat and embedding models.

        Args:
            chat: The chat model instance to use for text generation.
            embedding: The embedding model instance to use for text vectorization.

        Example:
            >>> from beeai_framework.adapters.openai import OpenAIChatModel, OpenAIEmbeddingModel
            >>> backend = Backend(
            ...     chat=OpenAIChatModel("gpt-4"),
            ...     embedding=OpenAIEmbeddingModel("text-embedding-3-small")
            ... )
        """
        self.chat = chat
        self.embedding = embedding

    @staticmethod
    def from_name(*, chat: str | ProviderName, embedding: str | ProviderName) -> "Backend":
        """Create a Backend instance from provider and model names.

        This factory method allows you to instantiate a Backend by specifying
        the provider and model names as strings, without needing to import
        specific model classes.

        Args:
            chat: The chat model identifier in the format "provider:model" or just "provider".
                Examples: "openai:gpt-4", "anthropic:claude-3-opus", "ollama".
            embedding: The embedding model identifier in the format "provider:model" or just "provider".
                Examples: "openai:text-embedding-3-small", "ollama:nomic-embed-text".

        Returns:
            A new Backend instance with the specified chat and embedding models.

        Example:
            >>> backend = Backend.from_name(
            ...     chat="openai:gpt-4",
            ...     embedding="openai:text-embedding-3-small"
            ... )
            >>> backend = Backend.from_name(
            ...     chat="anthropic:claude-3-opus",
            ...     embedding="ollama:nomic-embed-text"
            ... )
        """
        return Backend(chat=ChatModel.from_name(chat), embedding=EmbeddingModel.from_name(embedding))

    @staticmethod
    def from_provider(name: str | ProviderName) -> "Backend":
        """Create a Backend instance using the same provider for both models.

        This is a convenience method for when you want to use the same provider
        for both chat and embedding models. It uses the provider's default models.

        Args:
            name: The provider name (e.g., "openai", "anthropic", "ollama").
                The provider's default chat and embedding models will be used.

        Returns:
            A new Backend instance with both chat and embedding models from the same provider.

        Example:
            >>> # Uses OpenAI's default chat and embedding models
            >>> backend = Backend.from_provider("openai")
            
            >>> # Uses Ollama's default chat and embedding models
            >>> backend = Backend.from_provider("ollama")
        """
        return Backend.from_name(chat=name, embedding=name)

    async def clone(self) -> "Backend":
        """Create a deep copy of this Backend instance.

        This method clones both the chat and embedding models, creating
        independent copies that can be modified without affecting the original.

        Returns:
            A new Backend instance with cloned chat and embedding models.

        Example:
            >>> original = Backend.from_provider("openai")
            >>> cloned = await original.clone()
            >>> # Modifications to cloned won't affect original
        """
        return Backend(chat=await self.chat.clone(), embedding=await self.embedding.clone())
