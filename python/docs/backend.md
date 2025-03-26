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
  - [Embedding Model Configuration](#embedding-model-initialization)
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
| `OpenAI`         |  ✅  |          | `openai`     | OPENAI_CHAT_MODEL<br/>OPENAI_API_BASE<br/>OPENAI_API_KEY<br/>OPENAI_ORGANIZATION<br>OPENAI_API_HEADERS                                                                                                       |
| `Watsonx`        |  ✅  |          | `@ibm-cloud/watsonx-ai`  | WATSONX_CHAT_MODEL<br>WATSONX_API_KEY<br>WATSONX_PROJECT_ID<br>WATSONX_SPACE_ID<br>WATSONX_TOKEN<br>WATSONX_ZENAPIKEY<br>WATSONX_URL<br>WATSONX_REGION |
| `Groq`           |  ✅  |         | | GROQ_CHAT_MODEL<br>GROQ_API_KEY |
| `Amazon Bedrock` |  ✅  |         |  `boto3`| AWS_CHAT_MODEL<br>AWS_ACCESS_KEY_ID<br>AWS_SECRET_ACCESS_KEY<br>AWS_REGION<br>AWS_API_HEADERS |
| `Google Vertex`  |  ✅  |         |  | GOOGLE_VERTEX_CHAT_MODEL<br>GOOGLE_VERTEX_PROJECT<br>GOOGLE_APPLICATION_CREDENTIALS<br>GOOGLE_APPLICATION_CREDENTIALS_JSON<br>GOOGLE_CREDENTIALS<br>GOOGLE_VERTEX_API_HEADERS |
| `Azure OpenAI`   |  ✅  |         |  | AZURE_OPENAI_CHAT_MODEL<br>AZURE_OPENAI_API_KEY<br>AZURE_OPENAI_API_BASE<br>AZURE_OPENAI_API_VERSION<br>AZURE_AD_TOKEN<br>AZURE_API_TYPE<br>AZURE_API_HEADERS |
| `Anthropic`      |  ✅  |         |  | ANTHROPIC_CHAT_MODEL<br>ANTHROPIC_API_KEY<br>ANTHROPIC_API_HEADERS |
| `xAI`           |  ✅  |         | | XAI_CHAT_MODEL<br>XAI_API_KEY |

> [!TIP]
>
> If you don't see your provider raise an issue [here](https://github.com/i-am-bee/beeai-framework/discussions). Meanwhile, you can use [Ollama adapter](/python/examples/backend/providers/ollama.py).

---

### Backend initialization

The `Backend` class serves as a central entry point to access models from your chosen provider.

<!-- embedme examples/backend/providers/watsonx.py -->

```py
import asyncio
import json
import sys
import traceback

from pydantic import BaseModel, Field

from beeai_framework import ToolMessage
from beeai_framework.adapters.watsonx.backend.chat import WatsonxChatModel
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import MessageToolResultContent, UserMessage
from beeai_framework.cancellation import AbortSignal
from beeai_framework.errors import AbortError, FrameworkError
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool

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


async def watsonx_images() -> None:
    image_llm = ChatModel.from_name(
        "watsonx:meta-llama/llama-3-2-11b-vision-instruct",
    )
    response = await image_llm.create(
        messages=[
            UserMessage("What is the dominant color in the picture?"),
            UserMessage.from_image(
                "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAHUlEQVR4nGI5Y6bFQApgIkn1qIZRDUNKAyAAAP//0ncBT3KcmKoAAAAASUVORK5CYII="
            ),
        ],
    )
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


async def watson_tool_calling() -> None:
    watsonx_llm = ChatModel.from_name(
        "watsonx:ibm/granite-3-8b-instruct",
    )
    user_message = UserMessage("What is the current weather in Boston?")
    weather_tool = OpenMeteoTool()
    response = await watsonx_llm.create(messages=[user_message], tools=[weather_tool])
    tool_call_msg = response.get_tool_calls()[0]
    print(tool_call_msg.model_dump())
    tool_response = await weather_tool.run(json.loads(tool_call_msg.args))
    tool_response_msg = ToolMessage(
        MessageToolResultContent(
            result=tool_response.get_text_content(), tool_name=tool_call_msg.tool_name, tool_call_id=tool_call_msg.id
        )
    )
    print(tool_response_msg.to_plain())
    final_response = await watsonx_llm.create(messages=[user_message, tool_response_msg], tools=[])
    print(final_response.get_text_content())


async def watsonx_debug() -> None:
    # Log every request
    llm.emitter.match(
        "*",
        lambda data, event: print(
            f"Time: {event.created_at.time().isoformat()}",
            f"Event: {event.name}",
            f"Data: {str(data)[:90]}...",
        ),
    )

    response = await llm.create(
        messages=[UserMessage("Hello world!")],
    )
    print(response.messages[0].to_plain())


async def main() -> None:
    print("*" * 10, "watsonx_from_name")
    await watsonx_from_name()
    print("*" * 10, "watsonx_images")
    await watsonx_images()
    print("*" * 10, "watsonx_sync")
    await watsonx_sync()
    print("*" * 10, "watsonx_stream")
    await watsonx_stream()
    print("*" * 10, "watsonx_stream_abort")
    await watsonx_stream_abort()
    print("*" * 10, "watson_structure")
    await watson_structure()
    print("*" * 10, "watson_tool_calling")
    await watson_tool_calling()
    print("*" * 10, "watsonx_debug")
    await watsonx_debug()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

_Source: [examples/backend/providers/watsonx.py](/python/examples/backend/providers/watsonx.py)_

All providers examples can be found in [examples/backend/providers](/python/examples/backend/providers).

> [!TIP]
> See the [events documentation](/python/docs/events.md) for more information on standard emitter events.

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

<!-- embedme examples/backend/chat.py -->

```python
import asyncio
import sys
import traceback

from beeai_framework import UserMessage
from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.errors import FrameworkError
from examples.helpers.io import ConsoleReader


async def main() -> None:
    llm = OllamaChatModel("llama3.1")

    #  Optionally one may set llm parameters
    llm.parameters.max_tokens = 10000  # high number yields longer potential output
    llm.parameters.top_p = 0  # higher number yields more complex vocabulary, recommend only changing p or k
    llm.parameters.frequency_penalty = 0  # higher number yields reduction in word reptition
    llm.parameters.temperature = 0  # higher number yields greater randomness and variation
    llm.parameters.top_k = 0  # higher number yields more variance, recommend only changing p or k
    llm.parameters.n = 1  # higher number yields more choices
    llm.parameters.presence_penalty = 0  # higher number yields reduction in repetition of words
    llm.parameters.seed = 10  # can help produce similar responses if prompt and seed are always the same
    llm.parameters.stop_sequences = ["q", "quit", "ahhhhhhhhh"]  # stops the model on input of any of these strings
    llm.parameters.stream = False  # determines whether or not to use streaming to receive incremental data

    reader = ConsoleReader()

    for prompt in reader:
        response = await llm.create(messages=[UserMessage(prompt)])
        reader.write("LLM 🤖 (txt) : ", response.get_text_content())
        reader.write("LLM 🤖 (raw) : ", "\n".join([str(msg.to_plain()) for msg in response.messages]))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

_Source: [examples/backend/chat.py](/python/examples/backend/chat.py)_

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

<!-- embedme examples/backend/structured.py -->

```py
import asyncio
import json
import sys
import traceback

from pydantic import BaseModel, Field

from beeai_framework import UserMessage
from beeai_framework.backend.chat import ChatModel
from beeai_framework.errors import FrameworkError


async def main() -> None:
    model = ChatModel.from_name("ollama:llama3.1")

    class ProfileSchema(BaseModel):
        first_name: str = Field(..., min_length=1)
        last_name: str = Field(..., min_length=1)
        address: str
        age: int = Field(..., min_length=1)
        hobby: str

    class ErrorSchema(BaseModel):
        error: str

    class SchemUnion(ProfileSchema, ErrorSchema):
        pass

    response = await model.create_structure(
        schema=SchemUnion,
        messages=[UserMessage("Generate a profile of a citizen of Europe.")],
    )

    print(
        json.dumps(
            response.object.model_dump() if isinstance(response.object, BaseModel) else response.object, indent=4
        )
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

_Source: /examples/backend/structured.py_

### Tool calling

Integrate external tools with your AI model:

<!-- embedme examples/backend/tool_calling.py -->

```py
import asyncio
import json
import re
import sys
import traceback

from beeai_framework import SystemMessage, ToolMessage, UserMessage
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AnyMessage, MessageToolResultContent
from beeai_framework.backend.types import ChatModelParameters
from beeai_framework.errors import FrameworkError
from beeai_framework.tools import ToolOutput
from beeai_framework.tools.search import DuckDuckGoSearchTool
from beeai_framework.tools.tool import AnyTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool


async def main() -> None:
    model = ChatModel.from_name("ollama:llama3.1", ChatModelParameters(temperature=0))
    tools: list[AnyTool] = [DuckDuckGoSearchTool(), OpenMeteoTool()]
    messages: list[AnyMessage] = [
        SystemMessage("You are a helpful assistant. Use tools to provide a correct answer."),
        UserMessage("What's the fastest marathon time?"),
    ]

    while True:
        response = await model.create(
            messages=messages,
            tools=tools,
        )

        tool_calls = response.get_tool_calls()

        tool_results: list[ToolMessage] = []

        for tool_call in tool_calls:
            print(f"-> running '{tool_call.tool_name}' tool with {tool_call.args}")
            tool: AnyTool = next(tool for tool in tools if tool.name == tool_call.tool_name)
            assert tool is not None
            res: ToolOutput = await tool.run(json.loads(tool_call.args))
            result = res.get_text_content()
            print(f"<- got response from '{tool_call.tool_name}'", re.sub(r"\s+", " ", result)[:90] + " (truncated)")
            tool_results.append(
                ToolMessage(
                    MessageToolResultContent(
                        result=result,
                        tool_name=tool_call.tool_name,
                        tool_call_id=tool_call.id,
                    )
                )
            )

        messages.extend(tool_results)

        answer = response.get_text_content()

        if answer:
            print(f"Agent: {answer}")
            break


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

_Source: /examples/backend/tool_calling.py_

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
