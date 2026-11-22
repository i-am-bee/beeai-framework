# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
from typing import Any
from unittest.mock import MagicMock

import pytest

from beeai_framework.backend import ChatModel, ChatModelParameters, UserMessage

# The adapter's runtime dependencies are not installed with beeai-framework.
for module in ("torch", "peft", "outlines", "transformers"):
    pytest.importorskip(module, reason=f"{module} is not installed")

import torch  # noqa: E402

from beeai_framework.adapters.transformers.backend.chat import TransformersChatModel  # noqa: E402


def _make_model(parameters: ChatModelParameters | None = None) -> tuple[TransformersChatModel, MagicMock]:
    model = TransformersChatModel.__new__(TransformersChatModel)
    ChatModel.__init__(model, parameters=parameters or ChatModelParameters())
    model._model_id = "test/model"
    model.tokenizer = MagicMock()
    model.tokenizer.apply_chat_template.return_value = {"input_ids": torch.zeros((1, 3), dtype=torch.long)}
    model.tokenizer.encode.return_value = [1]
    model.tokenizer.decode.return_value = "ok"
    model._device_first_layer = torch.device("cpu")
    generate = MagicMock(return_value=torch.zeros((1, 5), dtype=torch.long))
    model._model = MagicMock(generate=generate)
    return model, generate


async def _generate_kwargs(model: TransformersChatModel, generate: MagicMock, **options: Any) -> dict[str, Any]:
    await model.run([UserMessage("hi")], **options)
    return generate.call_args.kwargs


@pytest.mark.asyncio
async def test_model_level_parameters_are_used() -> None:
    model, generate = _make_model(
        ChatModelParameters(
            temperature=0.7, top_p=0.8, top_k=40, max_tokens=64, stop_sequences=["END"], frequency_penalty=0.5
        )
    )
    kwargs = await _generate_kwargs(model, generate)

    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.8
    assert kwargs["top_k"] == 40
    assert kwargs["max_new_tokens"] == 64
    assert kwargs["frequency_penalty"] == 0.5
    assert kwargs["do_sample"] is True
    assert len(kwargs["stopping_criteria"]) == 1


@pytest.mark.asyncio
async def test_call_parameters_override_model_level_ones() -> None:
    model, generate = _make_model(ChatModelParameters(temperature=0.7, max_tokens=64, stop_sequences=["END"]))
    kwargs = await _generate_kwargs(model, generate, temperature=0, max_tokens=8, stop_sequences=["A", "B"])

    assert kwargs["temperature"] == 0
    assert kwargs["max_new_tokens"] == 8
    assert len(kwargs["stopping_criteria"]) == 2


@pytest.mark.asyncio
async def test_defaults_without_parameters() -> None:
    model, generate = _make_model()
    kwargs = await _generate_kwargs(model, generate)

    assert kwargs["temperature"] == 0
    assert kwargs["top_p"] is None
    assert kwargs["max_new_tokens"] is None
    assert kwargs["do_sample"] is False
    assert kwargs["stopping_criteria"] == []
