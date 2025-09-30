# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
from collections.abc import AsyncGenerator
from typing import Any, Unpack

import outlines
from outlines.inputs import Chat
from outlines.types import JsonSchema
from peft import PeftModel
from pydantic import BaseModel
from transformers import (  # type: ignore [attr-defined]
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    TextIteratorStreamer,
    set_seed,
)

from beeai_framework.adapters.litellm.utils import to_strict_json_schema
from beeai_framework.adapters.transformers.backend._utils import (
    CustomStoppingCriteria,
    get_do_sample,
    get_num_beams,
)
from beeai_framework.backend.chat import ChatModel, ChatModelKwargs
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.message import (
    AssistantMessage,
    MessageTextContent,
    ToolMessage,
)
from beeai_framework.backend.types import (
    ChatModelInput,
    ChatModelOutput,
)
from beeai_framework.backend.utils import parse_broken_json
from beeai_framework.context import RunContext
from beeai_framework.logger import Logger
from beeai_framework.tools.tool import AnyTool, Tool
from beeai_framework.utils.dicts import (
    exclude_none,
)
from beeai_framework.utils.models import is_pydantic_model
from beeai_framework.utils.strings import to_json

logger = Logger(__name__)


class TransformersChatModel(ChatModel):
    """
    Represents a Transformers-based chat model for processing and responding to conversational inputs.

    This class is built upon the Hugging Face Transformers library and integrates with a local
    pre-trained language models and tokenizer utilities. It is designed to handle conversion of
    structured chat input into the language model's input format, generate responses, and optionally
    support streaming of the output for applications requiring incremental response delivery.
    The class also allows for the integration of QLoRA adapters to enable fine-tuned models.
    """

    def __init__(
        self,
        model_id: str | None = None,
        *,
        qlora_adapter_id: str | None = None,
        hf_token: str | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        qlora_kwargs: dict[str, Any] | None = None,
        **kwargs: Unpack[ChatModelKwargs],
    ) -> None:
        super().__init__(**kwargs)
        hf_token = hf_token or os.getenv("HF_TOKEN", None)
        self._model_id = (
            model_id if model_id else os.getenv("TRANSFORMERS_CHAT_MODEL", "ibm-granite/granite-3.3-8b-instruct")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, **(tokenizer_kwargs or {}))  # type: ignore
        model_base = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            device_map="auto",
            token=hf_token,
            **(model_kwargs or {}),
        )
        self._model: Any = (
            model_base
            if qlora_adapter_id is None
            else PeftModel.from_pretrained(
                model_base, qlora_adapter_id, device_map="auto", token=hf_token, **(qlora_kwargs or {})
            )
        )
        self._model.eval()
        self._model_structured = outlines.from_transformers(self._model, self.tokenizer)  # type: ignore

        first_layer_name = next(iter(self._model.hf_device_map.keys()))
        self._device_first_layer = self._model.hf_device_map[first_layer_name]

    @property
    def model_id(self) -> str:
        """The ID for Causal Language Model at https://huggingface.co/models."""
        return self._model_id

    @property
    def provider_id(self) -> ProviderName:
        return "transformers"

    async def _create(
        self,
        input: ChatModelInput,
        run: RunContext,
    ) -> ChatModelOutput:
        model_output, model_output_structured = await self._get_model_output(input, streamer=None)
        logger.debug(f"Inference response output:\n{model_output}")
        return ChatModelOutput(output=[AssistantMessage(model_output)], output_structured=model_output_structured)

    async def _create_stream(
        self,
        input: ChatModelInput,
        run: RunContext,
    ) -> AsyncGenerator[ChatModelOutput]:
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        _, model_output_structured = await self._get_model_output(input, streamer)

        for chunk in streamer:
            if len(chunk) > 0:
                yield ChatModelOutput(output=[AssistantMessage(chunk)])

        if input.response_format:
            yield ChatModelOutput(output=[], output_structured=model_output_structured)

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
                # TODO: might incorrectly handle some non-text messages
                messages.append({"role": message.role, "content": message.text})

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

        return {
            "model": f"{self.provider_id}/{self.model_id}",
            "messages": messages,
            "tools": tools if tools else None,
            "response_format": (self._format_response_model(input.response_format) if input.response_format else None),
            "max_retries": 0,
            "tool_choice": tool_choice if tools else None,
            "parallel_tool_calls": (bool(input.parallel_tool_calls) if tools else None),
        }

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

    async def _get_model_output(
        self, input: ChatModelInput, streamer: TextIteratorStreamer | None
    ) -> tuple[str, Any | None]:
        llm_input = self._transform_input(input)
        inputs = self.tokenizer.apply_chat_template(
            llm_input["messages"],
            tools=llm_input["tools"],
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        prompt_tokens = inputs["input_ids"].shape[1]
        inputs_on_device = {k: v.to(self._device_first_layer) for k, v in inputs.items()}

        if input.seed is not None:
            set_seed(input.seed)

        kwargs = {
            "streamer": streamer,
            "max_new_tokens": input.max_tokens,
            "temperature": input.temperature,
            "top_k": input.top_k,
            "top_p": input.top_p,
            "num_beams": get_num_beams(input),
            "frequency_penalty": input.frequency_penalty,
            "presence_penalty": input.presence_penalty,
            "do_sample": get_do_sample(input),
            "stopping_criteria": self._get_stopping_criteria(input, prompt_tokens),
        }
        if input.response_format:
            generator = outlines.Generator(
                self._model_structured,
                JsonSchema(
                    input.response_format
                    if isinstance(input.response_format, dict)
                    else input.response_format.model_json_schema(mode="serialization")
                ),
            )
            model_output = await asyncio.to_thread(
                generator,  # type: ignore
                Chat(llm_input["messages"]),
                **kwargs,
            )
            model_parsed_output = parse_broken_json(model_output)
            if is_pydantic_model(input.response_format):
                model_parsed_output = input.response_format.model_validate(model_parsed_output)
            return model_output, model_parsed_output
        else:
            model_output = await asyncio.to_thread(
                self._model.generate,
                **inputs_on_device,
                **kwargs,
            )
            generated_tokens = model_output[0, prompt_tokens:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return generated_text, None

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
