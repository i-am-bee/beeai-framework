# Backend

*Disclaimer: The notes below may refer to the TypeScript version or missing files as the Python version moves toward parity in the near future. Additional Python examples coming soon. TODO*

> [!TIP]
>
> Location for concrete implementations within the framework `beeai/adapters/provider/backend`.
>
> Location for base abstraction within the framework `beeai/backend`.

The backend module is an umbrella module that encapsulates a unified way to work with the following functionalities:

- Chat Models via (`ChatModel` class)
- Embedding Models via (`EmbeddingModel` class)
- Audio Models (coming soon)
- Image Models (coming soon)



## Providers (implementations)

The following table depicts supported providers.

| Name             | Chat | Embedding | Dependency               | Environment Variables                                                                                                                                                 |
| ---------------- | :--: | :-------: | ------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Ollama`         |  ✅  |    ✅     | `ollama-ai-provider`     | OLLAMA_CHAT_MODEL<br>OLLAMA_EMBEDDING_MODEL<br/>OLLAMA_BASE_URL                                                                                                       |
| `Watsonx`        |  ✅  |    ✅     | `@ibm-cloud/watsonx-ai`  | WATSONX_CHAT_MODEL<br/>WATSONX_EMBEDDING_MODEL<br>WATSONX_API_KEY<br/>WATSONX_PROJECT_ID<br/>WATSONX_SPACE_ID<br>WATSONX_VERSION<br>WATSONX_REGION                    |

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

```txt
Coming soon
```

or you can always create the concrete provider's chat model directly

```txt
Coming soon
```

### Configuration

```txt
Coming soon
```

### Generation

```txt
Coming soon
```

> [!NOTE]
>
> Execution parameters (those passed to `model.create({...})`) are superior to ones defined via `config`.

### Stream

```txt
Coming soon
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
