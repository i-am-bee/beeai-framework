# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import cached_property
from typing import Any, Self

from pydantic import ConfigDict, TypeAdapter
from typing_extensions import TypedDict, Unpack

from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.errors import EmbeddingModelError
from beeai_framework.backend.events import (
    EmbeddingModelErrorEvent,
    EmbeddingModelStartEvent,
    EmbeddingModelSuccessEvent,
    embedding_model_event_types,
)
from beeai_framework.backend.types import EmbeddingModelInput, EmbeddingModelOutput
from beeai_framework.backend.utils import load_model, parse_model
from beeai_framework.context import Run, RunContext, RunMiddlewareType
from beeai_framework.emitter import Emitter
from beeai_framework.retryable import Retryable, RetryableConfig, RetryableInput
from beeai_framework.utils import AbortSignal
from beeai_framework.utils.dicts import exclude_non_annotated


class EmbeddingModelKwargs(TypedDict, total=False):
    """Configuration options for initializing an EmbeddingModel.

    This TypedDict defines all the optional keyword arguments that can be passed
    to an EmbeddingModel constructor to customize its behavior.
    """

    middlewares: Sequence[RunMiddlewareType]
    """
    List of middleware to apply during model execution.
    """

    settings: dict[str, Any]
    """
    Additional provider-specific settings.
    """

    __pydantic_config__ = ConfigDict(extra="forbid", arbitrary_types_allowed=True)  # type: ignore


_EmbeddingModelKwargsAdapter = TypeAdapter(EmbeddingModelKwargs)


class EmbeddingModel(ABC):
    """Abstract base class for all embedding model implementations.

    EmbeddingModel provides a unified interface for converting text into vector
    embeddings using various providers (OpenAI, Ollama, Cohere, etc.). It handles
    batching, retries, error handling, and event emission.

    The class is designed to be subclassed by provider-specific implementations
    that implement the `_create` abstract method.

    Attributes:
        middlewares: List of middleware functions to apply during execution.

    Example:
        >>> from beeai_framework.adapters.openai import OpenAIEmbeddingModel
        >>>
        >>> # Create an embedding model
        >>> model = OpenAIEmbeddingModel("text-embedding-3-small")
        >>>
        >>> # Generate embeddings for text
        >>> result = await model.create(["Hello, world!", "How are you?"])
        >>> print(len(result.embeddings))  # 2
        >>> print(len(result.embeddings[0]))  # 1536 (dimension size)
        >>>
        >>> # Use with retry
        >>> result = await model.create(
        ...     ["Text to embed"],
        ...     max_retries=3
        ... )
        >>>
        >>> # Create from name
        >>> model = EmbeddingModel.from_name("openai:text-embedding-3-small")
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """The unique identifier for this model (e.g., 'text-embedding-3-small')."""
        pass

    @property
    @abstractmethod
    def provider_id(self) -> ProviderName:
        """The provider name for this model (e.g., 'openai', 'ollama')."""
        pass

    @cached_property
    def emitter(self) -> Emitter:
        """Get the event emitter for this embedding model.

        Returns:
            An Emitter instance configured for embedding model events.
        """
        return self._create_emitter()

    def _create_emitter(self) -> Emitter:
        """Create an event emitter for this embedding model.

        Returns:
            An Emitter instance configured with the appropriate namespace and events.
        """
        return Emitter.root().child(
            namespace=["backend", self.provider_id, "embedding"],
            creator=self,
            events=embedding_model_event_types,
        )

    def __init__(self, **kwargs: Unpack[EmbeddingModelKwargs]) -> None:
        """Initialize an EmbeddingModel with the given configuration.

        Args:
            **kwargs: Configuration options as defined in EmbeddingModelKwargs.
                See EmbeddingModelKwargs documentation for all available options.

        Example:
            >>> from beeai_framework.adapters.openai import OpenAIEmbeddingModel
            >>> model = OpenAIEmbeddingModel(
            ...     "text-embedding-3-small",
            ...     settings={"dimensions": 512}
            ... )
        """
        self._settings: dict[str, Any] = kwargs.get("settings", {})
        self._settings.update(**exclude_non_annotated(kwargs, EmbeddingModelKwargs))

        kwargs = _EmbeddingModelKwargsAdapter.validate_python(kwargs)
        self.middlewares: list[RunMiddlewareType] = [*kwargs.get("middlewares", [])]

    def create(
        self, values: list[str], *, signal: AbortSignal | None = None, max_retries: int | None = None
    ) -> Run[EmbeddingModelOutput]:
        """Generate embeddings for a list of text strings.

        This method converts text strings into vector embeddings that can be used
        for semantic search, similarity comparison, clustering, and other ML tasks.

        Args:
            values: List of text strings to convert into embeddings.
            signal: Optional abort signal to cancel the operation.
            max_retries: Maximum number of retry attempts on failure. Defaults to 0.

        Returns:
            A Run object that yields EmbeddingModelOutput containing the embeddings
            and usage information.

        Raises:
            EmbeddingModelError: If the embedding generation fails.

        Example:
            >>> model = EmbeddingModel.from_name("openai:text-embedding-3-small")
            >>>
            >>> # Generate embeddings
            >>> result = await model.create([
            ...     "The quick brown fox",
            ...     "jumps over the lazy dog"
            ... ])
            >>> print(len(result.embeddings))  # 2
            >>> print(len(result.embeddings[0]))  # 1536
            >>>
            >>> # With retry
            >>> result = await model.create(
            ...     ["Text to embed"],
            ...     max_retries=3
            ... )
            >>>
            >>> # With abort signal
            >>> from beeai_framework.utils import AbortController
            >>> controller = AbortController()
            >>> result = await model.create(
            ...     ["Text to embed"],
            ...     signal=controller.signal
            ... )
        """
        model_input = EmbeddingModelInput(values=values, signal=signal, max_retries=max_retries or 0)

        async def handler(context: RunContext) -> EmbeddingModelOutput:
            try:
                await context.emitter.emit("start", EmbeddingModelStartEvent(input=model_input))

                result = await Retryable(
                    RetryableInput(
                        executor=lambda _: self._create(model_input, context),
                        config=RetryableConfig(
                            max_retries=(
                                model_input.max_retries
                                if model_input is not None and model_input.max_retries is not None
                                else 0
                            ),
                            signal=context.signal,
                        ),
                    )
                ).get()

                await context.emitter.emit("success", EmbeddingModelSuccessEvent(value=result))
                return result
            except Exception as ex:
                error = EmbeddingModelError.ensure(ex, model=self)
                await context.emitter.emit("error", EmbeddingModelErrorEvent(input=model_input, error=error))
                raise error
            finally:
                await context.emitter.emit("finish", None)

        return RunContext.enter(self, handler, signal=signal, run_params=model_input.model_dump()).middleware(
            *self.middlewares
        )

    @staticmethod
    def from_name(name: str | ProviderName, **kwargs: Any) -> "EmbeddingModel":
        """Create an EmbeddingModel instance from a provider and model name.

        This factory method allows you to instantiate an embedding model by specifying
        the provider and model name as a string, without needing to import
        provider-specific classes.

        Args:
            name: The model identifier in the format "provider:model" or just "provider".
                Examples: "openai:text-embedding-3-small", "ollama:nomic-embed-text"
            **kwargs: Additional keyword arguments passed to the model constructor.
                See EmbeddingModelKwargs for available options.

        Returns:
            An EmbeddingModel instance of the appropriate provider-specific subclass.

        Example:
            >>> # Create with just model name
            >>> model = EmbeddingModel.from_name("openai:text-embedding-3-small")
            >>>
            >>> # Create with provider default
            >>> model = EmbeddingModel.from_name("openai")
            >>>
            >>> # Create with additional options
            >>> model = EmbeddingModel.from_name(
            ...     "openai:text-embedding-3-small",
            ...     settings={"dimensions": 512}
            ... )
        """
        parsed_model = parse_model(name)
        TargetChatModel: type = load_model(parsed_model.provider_id, "embedding")  # noqa: N806
        return TargetChatModel(parsed_model.model_id, **kwargs)  # type: ignore

    @abstractmethod
    async def _create(
        self,
        input: EmbeddingModelInput,
        run: RunContext,
    ) -> EmbeddingModelOutput:
        """Generate embeddings for the given input (implementation method).

        This method must be implemented by subclasses to provide the actual
        embedding generation logic for the specific provider.

        Args:
            input: The prepared input containing text values and parameters.
            run: The execution context for this run.

        Returns:
            The embedding output containing vectors and usage information.

        Raises:
            NotImplementedError: If the subclass doesn't implement this method.
        """
        raise NotImplementedError

    async def clone(self) -> Self:
        """Create a deep copy of this EmbeddingModel instance.

        This method creates an independent copy of the model that can be
        modified without affecting the original. Subclasses should override
        this method to properly clone their specific state.

        Returns:
            A new EmbeddingModel instance with the same configuration.

        Note:
            The default implementation returns self and logs a warning.
            Provider-specific implementations should override this method
            to create proper clones.

        Example:
            >>> original = EmbeddingModel.from_name("openai:text-embedding-3-small")
            >>> cloned = await original.clone()
            >>> # Modifications to cloned won't affect original
        """
        if type(self).clone == EmbeddingModel.clone:
            logging.warning(f"EmbeddingModel ({type(self)!s}) does not implement the 'clone' method.")

        return self

    def destroy(self) -> None:
        """Clean up resources used by this embedding model.

        This method destroys the event emitter and releases any associated resources.
        Should be called when the model is no longer needed.

        Example:
            >>> model = EmbeddingModel.from_name("openai:text-embedding-3-small")
            >>> # Use the model...
            >>> model.destroy()
        """
        self.emitter.destroy()
