# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from pydantic import BaseModel

ProviderName = Literal[
    "ollama", "openai", "watsonx", "groq", "xai", "vertexai", "amazon_bedrock", "anthropic", "azure_openai", "mistralai"
]
ProviderHumanName = Literal[
    "Ollama", "OpenAI", "Watsonx", "Groq", "XAI", "VertexAI", "AmazonBedrock", "Anthropic", "AzureOpenAI", "MistralAI"
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
    "mistralAI": ProviderDef(name="MistralAI", module="mistralai", aliases=["mistral"]),
}
