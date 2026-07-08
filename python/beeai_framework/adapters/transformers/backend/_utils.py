# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
from typing import Any

import torch
from transformers import (
    StoppingCriteria,
)

from beeai_framework.backend.types import (
    ChatModelParameters,
)


def get_do_sample(input: ChatModelParameters) -> bool:
    return bool(
        input.temperature > 0.0
        or (input.top_k is not None and input.top_k > 1)
        or input.top_p is not None
        or (input.n is not None and input.n > 1)
    )


def get_num_beams(input: ChatModelParameters) -> int:
    return input.n if input.n is not None else 1


class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_token_ids: list[int], prompt_tokens: int) -> None:
        self.stop_token_ids = torch.tensor(stop_token_ids, dtype=torch.long)
        self.prompt_tokens = prompt_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        if input_ids.shape[1] <= self.prompt_tokens:
            return False

        stop_len = self.stop_token_ids.shape[0]
        if stop_len == 0:
            return False

        # Move tensor to the correct device on the first call
        if self.stop_token_ids.device != input_ids.device:
            self.stop_token_ids = self.stop_token_ids.to(input_ids.device)

        # Check if the last `stop_len` tokens match the stop sequence
        return (input_ids.shape[1] >= (self.prompt_tokens + stop_len)) and torch.equal(
            input_ids[0, -stop_len:], self.stop_token_ids
        )
