# Copyright 2025 IBM Corp.
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


from typing import Literal

from pydantic import BaseModel

ProviderName = Literal[
    "ollama", "openai", "watsonx", "groq", "xai", "vertexai", "amazon_bedrock", "anthropic", "azure_openai"
]
ProviderHumanName = Literal[
    "Ollama", "OpenAI", "Watsonx", "Groq", "XAI", "VertexAI", "AmazonBedrock", "Anthropic", "AzureOpenAI"
]


class ProviderDef(BaseModel):
    name: ProviderHumanName
    module: ProviderName
    aliases: list[str]


class ProviderModelDef(BaseModel):
    provider_id: str
    model_id: str | None = None
    provider_def: ProviderDef


BackendProviders = {
    "Ollama": ProviderDef(name="Ollama", module="ollama", aliases=[]),
    "OpenAI": ProviderDef(name="OpenAI", module="openai", aliases=["openai"]),
    "watsonx": ProviderDef(name="Watsonx", module="watsonx", aliases=["watsonx", "ibm"]),
    "Groq": ProviderDef(name="Groq", module="groq", aliases=["groq"]),
    "xAI": ProviderDef(name="XAI", module="xai", aliases=["xai", "grok"]),
    "vertexAI": ProviderDef(name="VertexAI", module="vertexai", aliases=["vertexai", "google"]),
    "AmazonBedrock": ProviderDef(
        name="AmazonBedrock",
        module="amazon_bedrock",
        aliases=["amazon_bedrock", "amazon", "bedrock"],
    ),
    "anthropic": ProviderDef(name="Anthropic", module="anthropic", aliases=["anthropic"]),
    "AzureOpenAI": ProviderDef(
        name="AzureOpenAI",
        module="azure_openai",
        aliases=["azure_openai", "azure"],
    ),
}
