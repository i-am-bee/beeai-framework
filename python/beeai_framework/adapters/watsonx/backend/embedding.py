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


from beeai_framework.backend import EmbeddingModel, EmbeddingModelOutput
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.types import EmbeddingModelInput
from beeai_framework.context import RunContext


class WatsonxEmbeddingModel(EmbeddingModel):
    def __init__(self, model_id: str) -> None:
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider_id(self) -> ProviderName:
        return "watsonx"

    async def _create(self, input: EmbeddingModelInput, run: RunContext) -> EmbeddingModelOutput:
        raise NotImplementedError()
