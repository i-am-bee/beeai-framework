# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, TypeVar

try:
    from deepeval.key_handler import KEY_FILE_HANDLER, ModelKeyValues
    from deepeval.models import DeepEvalBaseLLM
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [evaluation] not found.\nRun 'pip install \"beeai-framework[evaluation]\"' to install."
    ) from e

from dotenv import load_dotenv
from pydantic import BaseModel

from beeai_framework.backend import ChatModel, ChatModelParameters
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.message import UserMessage
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.utils import ModelLike

TSchema = TypeVar("TSchema", bound=BaseModel)

load_dotenv()


class DeepEvalLLM(DeepEvalBaseLLM):
    def __init__(self, model: ChatModel, *args: Any, **kwargs: Any) -> None:
        self._model = model
        super().__init__(model.model_id, *args, **kwargs)

    # pyrefly: ignore [bad-override]
    def load_model(self, *args: Any, **kwargs: Any) -> None:
        return None

    # pyrefly: ignore [bad-override]
    def generate(self, prompt: str, schema: BaseModel | None = None) -> str:
        raise NotImplementedError()

    # pyrefly: ignore [bad-override]
    async def a_generate(self, prompt: str, schema: TSchema | None = None) -> str:
        input_msg = UserMessage(prompt)
        response = await self._model.run(
            [input_msg],
            response_format=schema.model_json_schema(mode="serialization") if schema is not None else None,
            stream=False,
            temperature=0,
        ).middleware(
            GlobalTrajectoryMiddleware(
                pretty=True,
                exclude_none=True,
                enabled=os.environ.get("EVAL_LOG_LLM_CALLS", "").lower() == "true",
            )
        )
        return response.get_text_content()

    # pyrefly: ignore [bad-override]
    def get_model_name(self) -> str:
        return f"{self._model.model_id} ({self._model.provider_id})"

    @staticmethod
    def from_name(
        name: str | ProviderName | None = None,
        options: ModelLike[ChatModelParameters] | None = None,
        **kwargs: Any,
    ) -> "DeepEvalLLM":
        name = name or KEY_FILE_HANDLER.fetch_data(ModelKeyValues.LOCAL_MODEL_NAME)
        if not name:
            raise ValueError(
                "No model name provided and none configured via `deepeval set-local-model`. "
                "Pass `name` explicitly (e.g. 'ollama:llama3.1:8b')."
            )
        model = ChatModel.from_name(name, options, **kwargs)
        return DeepEvalLLM(model)
