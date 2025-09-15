# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Annotated, Any, Self

from beeai_framework.utils.io import setup_io_context

try:
    from beeai_sdk.a2a.extensions import FormExtensionServer, FormExtensionSpec, FormRender, TextField
    from beeai_sdk.server.context import RunContext
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [beeai-platform] not found.\nRun 'pip install \"beeai-framework[beeai-platform]\"' to install."
    ) from e


class BeeAIPlatformIOContext:
    def __init__(
        self, context: RunContext, *, form: Annotated[FormExtensionServer, FormExtensionSpec(params=None)]
    ) -> None:
        self.context = context
        self._form = form
        self._cleanup: Callable[[], None] = lambda: None

    def __enter__(self) -> Self:
        self._cleanup = setup_io_context(read=self._read)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._cleanup()
        self._cleanup = lambda: None

    async def _read(self, prompt: str) -> str:
        answer_field_id = "answer"
        form_data = await self._form.request_form(
            form=FormRender(
                id="form",
                title=prompt,
                description="",
                columns=1,
                submit_label="Send",
                fields=[
                    TextField(
                        id=answer_field_id,
                        label="Answer",
                        required=True,
                        placeholder="",
                        type="text",
                        default_value="",
                        col_span=1,
                    )
                ],
            )
        )
        return str(form_data.values[answer_field_id].value)
