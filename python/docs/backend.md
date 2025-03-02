# ⚙️ Backend

<!-- TOC -->
## Table of Contents
- [Overview](#overview)
- [Supported Providers](#supported-providers) 
- [Backend Initialization](#backend-initialization)
- [Chat Model](#chat-model)
  - [Chat Model Configuration](#chat-model-configuration)
  - [Text Generation](#text-generation)
  - [Streaming Responses](#streaming-responses)
  - [Structured Generation](#structured-generation)
  - [Tool Calling](#tool-calling)
- [Embedding Model](#embedding-model)
  - [Embedding Model Configuration](#embedding-model-configuration)
  - [Embedding Model Usage](#embedding-model-usage)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)
<!-- /TOC -->

---

## Overview

Backend is an umbrella module that encapsulates a unified way to work with the following functionalities:

- Chat Models via (`ChatModel` class)
- Embedding Models (coming soon)
- Audio Models (coming soon)
- Image Models (coming soon)

BeeAI framework's backend is designed with a provider-based architecture, allowing you to switch between different AI service providers while maintaining a consistent API.

> [!NOTE]
>
> Location within the framework: [beeai_framework/backend](/python/beeai_framework/backend).

--- 

## Supported providers

The following table depicts supported providers. Each provider requires specific configuration through environment variables. Ensure all required variables are set before initializing a provider.

| Name             | Chat | Embedding | Dependency               | Environment Variables                                                                                                                                                 |
| ---------------- | :--: | :-------: | ------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Ollama`         |  ✅  |          | `ollama-ai-provider`     | OLLAMA_CHAT_MODEL<br/>OLLAMA_BASE_URL                                                                                                       |
| `OpenAI`         |  ✅  |          | `openai`     | OPENAI_CHAT_MODEL<br/>OPENAI_API_BASE<br/>OPENAI_API_KEY<br/>OPENAI_ORGANIZATION                                                                                                       |
| `Watsonx`        |  ✅  |          | `@ibm-cloud/watsonx-ai`  | WATSONX_CHAT_MODEL<br/>WATSONX_EMBEDDING_MODEL<br>WATSONX_API_KEY<br/>WATSONX_PROJECT_ID<br/>WATSONX_SPACE_ID<br>WATSONX_VERSION<br>WATSONX_REGION                    |
| `Groq`           |    |         | Coming soon! | GROQ_CHAT_MODEL<br>GROQ_EMBEDDING_MODEL<br>GROQ_API_HOST<br>GROQ_API_KEY |
| `Amazon Bedrock` |    |         | Coming soon! | AWS_CHAT_MODEL<br>AWS_EMBEDDING_MODEL<br>AWS_ACCESS_KEY_ID<br>AWS_SECRET_ACCESS_KEY<br>AWS_REGION<br>AWS_SESSION_TOKEN |
| `Google Vertex`  |    |         | Coming soon! | GOOGLE_VERTEX_CHAT_MODEL<br>GOOGLE_VERTEX_EMBEDDING_MODEL<br>GOOGLE_VERTEX_PROJECT<br>GOOGLE_VERTEX_ENDPOINT<br>GOOGLE_VERTEX_LOCATION |
| `Azure OpenAI`   |    |         | Coming soon! | AZURE_OPENAI_CHAT_MODEL<br>AZURE_OPENAI_EMBEDDING_MODEL<br>AZURE_OPENAI_API_KEY<br>AZURE_OPENAI_API_ENDPOINT<br>AZURE_OPENAI_API_RESOURCE<br>AZURE_OPENAI_API_VERSION |
| `Anthropic`      |    |         | Coming soon! | ANTHROPIC_CHAT_MODEL<br>ANTHROPIC_EMBEDDING_MODEL<br>ANTHROPIC_API_KEY<br>ANTHROPIC_API_BASE_URL<br>ANTHROPIC_API_HEADERS |

> [!TIP]
>
> If you don't see your provider raise an issue [here](https://github.com/i-am-bee/beeai-framework/discussions). Meanwhile, you can use [Ollama adapter](/python/examples/backend/providers/ollama.py).

---

### Backend initialization

The `Backend` class serves as a central entry point to access models from your chosen provider.

<!-- embedme examples/backend/providers/watsonx.py -->

```py
import asyncio

from pydantic import BaseModel, Field

from beeai_framework.adapters.watsonx.backend.chat import WatsonxChatModel
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import UserMessage
from beeai_framework.cancellation import AbortSignal
from beeai_framework.errors import AbortError

# Setting can be passed here during initiation or pre-configured via environment variables
llm = WatsonxChatModel(
    "ibm/granite-3-8b-instruct",
    # settings={
    #     "project_id": "WATSONX_PROJECT_ID",
    #     "api_key": "WATSONX_API_KEY",
    #     "api_base": "WATSONX_API_URL",
    # },
)


async def watsonx_from_name() -> None:
    watsonx_llm = ChatModel.from_name(
        "watsonx:ibm/granite-3-8b-instruct",
        # {
        #     "project_id": "WATSONX_PROJECT_ID",
        #     "api_key": "WATSONX_API_KEY",
        #     "api_base": "WATSONX_API_URL",
        # },
    )
    user_message = UserMessage("what states are part of New England?")
    response = await watsonx_llm.create(messages=[user_message])
    print(response.get_text_content())


async def watsonx_sync() -> None:
    user_message = UserMessage("what is the capital of Massachusetts?")
    response = await llm.create(messages=[user_message])
    print(response.get_text_content())


async def watsonx_stream() -> None:
    user_message = UserMessage("How many islands make up the country of Cape Verde?")
    response = await llm.create(messages=[user_message], stream=True)
    print(response.get_text_content())


async def watsonx_stream_abort() -> None:
    user_message = UserMessage("What is the smallest of the Cape Verde islands?")

    try:
        response = await llm.create(messages=[user_message], stream=True, abort_signal=AbortSignal.timeout(0.5))

        if response is not None:
            print(response.get_text_content())
        else:
            print("No response returned.")
    except AbortError as err:
        print(f"Aborted: {err}")


async def watson_structure() -> None:
    class TestSchema(BaseModel):
        answer: str = Field(description="your final answer")

    user_message = UserMessage("How many islands make up the country of Cape Verde?")
    response = await llm.create_structure(schema=TestSchema, messages=[user_message])
    print(response.object)


async def main() -> None:
    print("*" * 10, "watsonx_from_name")
    await watsonx_from_name()
    print("*" * 10, "watsonx_sync")
    await watsonx_sync()
    print("*" * 10, "watsonx_stream")
    await watsonx_stream()
    print("*" * 10, "watsonx_stream_abort")
    await watsonx_stream_abort()
    print("*" * 10, "watson_structure")
    await watson_structure()


if __name__ == "__main__":
    asyncio.run(main())

```

_Source: [examples/backend/providers/watsonx.py](/python/examples/backend/providers/watsonx.py)_

All providers examples can be found in [examples/backend/providers](/examples/backend/providers).

---

## Chat model

The `ChatModel` class represents a Chat Large Language Model and provides methods for text generation, streaming responses, and more. You can initialize a chat model in multiple ways:

**Method 1: Using the generic factory method**

```python
from beeai_framework.backend.chat import ChatModel

ollama_chat_model = ChatModel.from_name("ollama:llama3.1")
```

**Method 2: Creating a specific provider model directly**

```python
from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel

ollama_chat_model = OllamaChatModel("llama3.1")
```

### Chat model configuration

You can configure various parameters for your chat model:

```txt
Coming soon
```

### Text generation

The most basic usage is to generate text responses:

```python
from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.backend.message import UserMessage

ollama_chat_model = OllamaChatModel("llama3.1")
response = await ollama_chat_model.create(
    messages=[UserMessage("what states are part of New England?")]
)

print(response.get_text_content())
```

> [!NOTE]
>
> Execution parameters (those passed to `model.create({...})`) are superior to ones defined via `config`.

### Streaming responses

For applications requiring real-time responses:

```python
from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.backend.message import UserMessage

llm = OllamaChatModel("llama3.1")
user_message = UserMessage("How many islands make up the country of Cape Verde?")
response = await llm.create(messages=[user_message], stream=True)
```

### Structured generation

Generate structured data according to a schema:

```txt
Coming soon
```

_Source: /examples/backend/structured.py_

### Tool calling

Integrate external tools with your AI model:

```txt
Coming soon
```

_Source: /examples/backend/toolCalling.py_

---

## Embedding model

The `EmbedingModel` class provides functionality for generating vector embeddings from text.

### Embedding model initialization

You can initialize an embedding model in multiple ways:

**Method 1: Using the generic factory method**

```txt
Coming soon
```

**Method 2: Creating a specific provider model directly**

```txt
Coming soon
```

### Embedding model usage

Generate embeddings for one or more text strings:

```txt
Coming soon
```

---

## Advanced usage

If your preferred provider isn't directly supported, you can use the LangChain adapter as a bridge. 

This allows you to leverage any provider that has LangChain compatibility.

```txt
Coming soon
```

_Source: /examples/backend/providers/langchain.py_

---

## Troubleshooting

Common issues and their solutions:

1. Authentication errors: Ensure all required environment variables are set correctly
2. Model not found: Verify that the model ID is correct and available for the selected provider

---

## Examples

- All backend examples can be found in [here](/python/examples/backend).