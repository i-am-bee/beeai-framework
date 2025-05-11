from typing import Any, TypeVar

from deepeval.key_handler import KEY_FILE_HANDLER, KeyValues
from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel

from beeai_framework.backend import ChatModel, ChatModelParameters
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.message import UserMessage
from beeai_framework.utils import ModelLike

TSchema = TypeVar("TSchema", bound=BaseModel)


class DeepEvalLLM(DeepEvalBaseLLM):  # type: ignore[misc]
    def __init__(self, model: ChatModel, *args: Any, **kwargs: Any) -> None:
        self._model = model
        super().__init__(model.model_id, *args, **kwargs)

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        return None

    def generate(self, prompt: str, schema: BaseModel | None = None) -> str:
        raise NotImplementedError()

    async def a_generate(self, prompt: str, schema: TSchema | None = None) -> str:
        input_msg = UserMessage(prompt)

        response = await self._model.create(
            messages=[input_msg],
            response_format=schema.model_json_schema(mode="serialization") if schema is not None else None,
            stream=False,
        )
        text = response.get_text_content()
        return schema.model_validate_json(text).model_dump_json() if schema else text

    def get_model_name(self) -> str:
        return f"{self._model.model_id} ({self._model.provider_id})"

    @staticmethod
    def from_name(
        name: str | ProviderName | None = None, options: ModelLike[ChatModelParameters] | None = None, **kwargs: Any
    ) -> "DeepEvalLLM":
        name = name or KEY_FILE_HANDLER.fetch_data(KeyValues.LOCAL_MODEL_NAME)
        model = ChatModel.from_name(name, options, **kwargs)
        return DeepEvalLLM(model)
