# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
from collections.abc import AsyncGenerator
from typing import Any, Unpack

import outlines
import torch
from outlines.types import JsonSchema
from peft import PeftModel
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    TextIteratorStreamer,
    TextStreamer,
    set_seed,
)

from beeai_framework.adapters.litellm.utils import to_strict_json_schema
from beeai_framework.adapters.transformers.backend._utils import (
    CustomStoppingCriteria,
    get_do_sample,
    get_num_beams,
    get_prompt_chat_history,
)
from beeai_framework.backend.chat import ChatModel, ChatModelKwargs, T
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.message import (
    AssistantMessage,
    MessageTextContent,
    ToolMessage,
)
from beeai_framework.backend.types import (
    ChatModelInput,
    ChatModelOutput,
    ChatModelStructureInput,
    ChatModelStructureOutput,
)
from beeai_framework.backend.utils import (
    parse_broken_json,
)
from beeai_framework.context import RunContext
from beeai_framework.logger import Logger
from beeai_framework.tools.tool import AnyTool, Tool
from beeai_framework.utils.dicts import (
    exclude_keys,
    exclude_none,
    include_keys,
)
from beeai_framework.utils.strings import to_json

logger = Logger(__name__)


class TransformersChatModel(ChatModel):
    model: Any
    model_structured: Any
    _model_id: str
    device_first_layer: torch.device
    tokenizer: Any
    streamer: TextStreamer

    @property
    def model_id(self) -> str:
        """The ID for Causal Language Model at https://huggingface.co/models."""
        return self._model_id

    @property
    def provider_id(self) -> ProviderName:
        return "transformers"

    def __init__(
        self,
        model_id: str,
        *,
        qlora_adapter_id: str | None = None,
        hf_token: str | None = None,
        **kwargs: Unpack[ChatModelKwargs],
    ) -> None:
        # TODO: handle quantization config
        super().__init__(**kwargs)
        if hf_token is None:
            hf_token = os.getenv("HF_TOKEN", None)
        self.model_supports_tool_calling = False
        self.allow_parallel_tool_calls = False
        self.ignore_parallel_tool_calls = True
        self.use_strict_tool_schema = True
        self._model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)  # type: ignore
        model_base = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            token=hf_token,
        )
        self.model = (  # use peft if qlora_adapter_id is provided
            model_base
            if qlora_adapter_id is None
            else PeftModel.from_pretrained(model_base, qlora_adapter_id, device_map="auto", token=hf_token)
        )
        self.model.eval()
        self.model_structured = outlines.from_transformers(self.model, self.tokenizer)

        first_layer_name = next(iter(self.model.hf_device_map.keys()))
        self.device_first_layer = self.model.hf_device_map[first_layer_name]
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

    async def _create(
        self,
        input: ChatModelInput,
        run: RunContext,
    ) -> ChatModelOutput:
        model_output, prompt_tokens = await self._get_model_output(input=input, stream=False)
        generated_tokens = model_output[0, prompt_tokens:]
        generated_text = self.tokenizer.decode(generated_tokens)
        logger.debug(f"Inference response output:\n{generated_text}")

        return ChatModelOutput(messages=[AssistantMessage(generated_text)])

    async def _create_stream(
        self,
        input: ChatModelInput,
        run: RunContext,
    ) -> AsyncGenerator[ChatModelOutput]:
        await self._get_model_output(input=input, stream=True)

        chunk: tuple[int, str]
        for chunk in enumerate(self.streamer):  # type: ignore
            if len(chunk[1]) > 0:
                yield ChatModelOutput(messages=[AssistantMessage(chunk[1])])

    async def _create_structure(
        self,
        input: ChatModelStructureInput[T],
        run: RunContext,
    ) -> ChatModelStructureOutput:
        json_schema: dict[str, Any] = (
            input.input_schema
            if isinstance(input.input_schema, dict)
            else input.input_schema.model_json_schema(mode="serialization")
        )
        llm_input = self._transform_input(ChatModelInput(messages=input.messages)) | {"stream": False}
        prompt_chat_history = get_prompt_chat_history(llm_input["messages"])
        generator = outlines.Generator(self.model_structured, JsonSchema(json_schema))

        if input.seed is not None:
            set_seed(input.seed)

        text_response = await asyncio.to_thread(
            generator,
            prompt_chat_history,
            max_new_tokens=input.max_tokens,
            temperature=input.temperature,
            top_k=input.top_k,
            top_p=input.top_p,
            num_beams=get_num_beams(input),
            frequency_penalty=input.frequency_penalty,
            presence_penalty=input.presence_penalty,
            do_sample=get_do_sample(input),
        )
        result = parse_broken_json(text_response)

        # TODO: validate result matches expected schema
        return ChatModelStructureOutput(object=result)

    def _transform_input(self, input: ChatModelInput) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        for message in input.messages:
            if isinstance(message, ToolMessage):
                for content in message.content:
                    new_msg = (
                        {
                            "tool_call_id": content.tool_call_id,
                            "role": "tool",
                            "name": content.tool_name,
                            "content": content.result,
                        }
                        if self.model_supports_tool_calling
                        else {
                            "role": "assistant",
                            "content": to_json(
                                {
                                    "tool_call_id": content.tool_call_id,
                                    "result": content.result,
                                },
                                indent=2,
                                sort_keys=False,
                            ),
                        }
                    )
                    messages.append(new_msg)

            elif isinstance(message, AssistantMessage):
                msg_text_content = [t.model_dump() for t in message.get_text_messages()]
                msg_tool_calls = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "arguments": call.args,
                            "name": call.tool_name,
                        },
                    }
                    for call in message.get_tool_calls()
                ]

                new_msg = (
                    {
                        "role": "assistant",
                        "content": msg_text_content or None,
                        "tool_calls": msg_tool_calls or None,
                    }
                    if self.model_supports_tool_calling
                    else {
                        "role": "assistant",
                        "content": [
                            *msg_text_content,
                            *[
                                MessageTextContent(text=to_json(t, indent=2, sort_keys=False)).model_dump()
                                for t in msg_tool_calls
                            ],
                        ]
                        or None,
                    }
                )

                messages.append(exclude_none(new_msg))
            else:
                messages.append(message.to_plain())

        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": self._format_tool_model(tool.input_schema),
                    "strict": self.use_strict_tool_schema,
                },
            }
            for tool in input.tools or []
        ]

        supported_params = [
            "stream",
            "frequency_penalty",
            "max_tokens",
            "presence_penalty",
            "n",
            "temperature",
            "top_p",
            "tools",
            "tool_choice",
        ]
        settings = exclude_keys(
            self._settings | input.model_dump(exclude_unset=True),
            {
                *supported_params,
                "abort_signal",
                "model",
                "messages",
                "tools",
                "supports_top_level_unions",
            },
        )
        params = include_keys(
            input.model_dump(exclude_none=True)  # get all parameters with default values
            | self._settings  # get constructor overrides
            | self.parameters.model_dump(exclude_unset=True)  # get default parameters
            | input.model_dump(exclude_none=True, exclude_unset=True),  # get custom manually set parameters
            set(supported_params),
        )

        tool_choice: dict[str, Any] | str | AnyTool | None = input.tool_choice
        if input.tool_choice == "none" and input.tool_choice not in self._tool_choice_support:
            tool_choice = None
            tools = []
        elif input.tool_choice == "auto" and input.tool_choice not in self._tool_choice_support:
            tool_choice = None
        elif isinstance(input.tool_choice, Tool) and "single" in self._tool_choice_support:
            tool_choice = {
                "type": "function",
                "function": {"name": input.tool_choice.name},
            }
        elif input.tool_choice not in self._tool_choice_support:
            tool_choice = None

        if input.response_format:
            tools = []
            tool_choice = None

        return exclude_none(
            exclude_none(settings)
            | exclude_none(params)
            | {
                "model": f"{self.provider_id}/{self.model_id}",
                "messages": messages,
                "tools": tools if tools else None,
                "response_format": (
                    self._format_response_model(input.response_format) if input.response_format else None
                ),
                "max_retries": 0,
                "tool_choice": tool_choice if tools else None,
                "parallel_tool_calls": (bool(input.parallel_tool_calls) if tools else None),
            }
        )

    def _get_inputs_on_device(self, input: ChatModelInput, stream: bool) -> tuple[dict[str, Any], int]:
        llm_input = self._transform_input(input) | {"stream": stream}
        inputs = self.tokenizer.apply_chat_template(
            llm_input["messages"],
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        prompt_tokens = inputs["input_ids"].shape[1]
        inputs_on_device = {k: v.to(self.device_first_layer) for k, v in inputs.items()}

        return inputs_on_device, prompt_tokens

    def _get_stopping_criteria(self, input: ChatModelInput, prompt_tokens: int) -> list[StoppingCriteria]:
        return (
            [
                CustomStoppingCriteria(
                    self.tokenizer.encode(stop_word, add_prefix_space=False),
                    prompt_tokens,
                )
                for stop_word in input.stop_sequences
            ]
            if input.stop_sequences is not None
            else []
        )

    async def _get_model_output(self, input: ChatModelInput, stream: bool) -> tuple[Any, int]:
        inputs_on_device, prompt_tokens = self._get_inputs_on_device(input=input, stream=stream)

        if input.seed is not None:
            set_seed(input.seed)

        model_output = await asyncio.to_thread(
            self.model.generate,
            **inputs_on_device,
            streamer=(self.streamer if stream else None),
            max_new_tokens=input.max_tokens,
            temperature=input.temperature,
            top_k=input.top_k,
            top_p=input.top_p,
            num_beams=get_num_beams(input),
            frequency_penalty=input.frequency_penalty,
            presence_penalty=input.presence_penalty,
            do_sample=get_do_sample(input),
            stopping_criteria=self._get_stopping_criteria(input, prompt_tokens),
        )

        return model_output, prompt_tokens

    def _format_tool_model(self, model: type[BaseModel]) -> dict[str, Any]:
        return to_strict_json_schema(model) if self.use_strict_tool_schema else model.model_json_schema()

    def _format_response_model(self, model: type[BaseModel] | dict[str, Any]) -> type[BaseModel] | dict[str, Any]:
        if isinstance(model, dict) and model.get("type") in [
            "json_schema",
            "json_object",
        ]:
            return model

        json_schema = (
            {
                "schema": (to_strict_json_schema(model) if self.use_strict_tool_schema else model),
                "name": "schema",
                "strict": self.use_strict_model_schema,
            }
            if isinstance(model, dict)
            else {
                "schema": (to_strict_json_schema(model) if self.use_strict_tool_schema else model.model_json_schema()),
                "name": model.__name__,
                "strict": self.use_strict_model_schema,
            }
        )

        return {"type": "json_schema", "json_schema": json_schema}
