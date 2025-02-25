# Backend

*Disclaimer: The notes below may refer to the TypeScript version or missing files as the Python version moves toward parity in the near future. Additional Python examples coming soon. TODO*

> [!TIP]
>
> Location for concrete implementations within the framework `beeai/adapters/provider/backend`.
>
> Location for base abstraction within the framework `beeai/backend`.

The backend module is an umbrella module that encapsulates a unified way to work with the following functionalities:

- Chat Models via (`ChatModel` class)
- Embedding Models via (coming soon)
- Audio Models (coming soon)
- Image Models (coming soon)



## Providers (implementations)

The following table depicts supported providers.

| Name             | Chat | Embedding | Dependency               | Environment Variables                                                                                                                                                 |
| ---------------- | :--: | :-------: | ------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Ollama`         |  ✅  |          | `ollama-ai-provider`     | OLLAMA_CHAT_MODEL<br/>OLLAMA_BASE_URL                                                                                                       |
| `OpenAI`         |  ✅  |          | `openai`     | OPENAI_CHAT_MODEL<br/>OPENAI_API_BASE<br/>OPENAI_API_KEY<br/>OPENAI_ORGANIZATION                                                                                                       |
| `Watsonx`        |  ✅  |          | `@ibm-cloud/watsonx-ai`  | WATSONX_CHAT_MODEL<br/>WATSONX_EMBEDDING_MODEL<br>WATSONX_API_KEY<br/>WATSONX_PROJECT_ID<br/>WATSONX_SPACE_ID<br>WATSONX_VERSION<br>WATSONX_REGION                    |

> [!TIP]
>
> If you don't see your provider raise an issue [here](https://github.com/i-am-bee/beeai-framework/discussions). Meanwhile, you can use [Ollama adapter](/examples/backend/providers/ollama.py).

### Initialization

```txt
Coming soon
```

All providers examples can be found in [examples/backend/providers](/examples/backend/providers).

## Chat Model

The `ChatModel` class represents a Chat Large Language Model and can be initiated in one of the following ways.

```python
from beeai_framework.backend.chat import ChatModel

ollama_chat_model = ChatModel.from_name("ollama:llama3.1")
```

or you can always create the concrete provider's chat model directly

```python
from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel

ollama_chat_model = OllamaChatModel("llama3.1")
```

### Configuration

```txt
Coming soon
```

### Generation

```python
from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.backend.message import UserMessage

ollama_chat_model = OllamaChatModel("llama3.1")
response = await llm.create({
    "messages": [UserMessage("what states are part of New England?")]
})

print(response.get_text_content())
```

> [!NOTE]
>
> Execution parameters (those passed to `model.create({...})`) are superior to ones defined via `config`.

### Stream

```python
from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.backend.message import UserMessage

llm = OllamaChatModel("llama3.1")
user_message = UserMessage("How many islands make up the country of Cape Verde?")
response = await llm.create({"messages": [user_message], "stream": True})
```

### Structured Generation

```py
```

Source: /examples/backend/structured.py TODO

### Tool Calling

```py
```

Source: /examples/backend/toolCalling.py TODO

## Embedding Model

The `EmbedingModel` class represents an Embedding Model and can be initiated in one of the following ways.

```
Coming soon
```

or you can always create the concrete provider's embedding model directly

```
Coming soon
```

### Usage

```txt
Coming soon
```
