# Copyright 2026 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing
from typing import Any

from sse_starlette import ServerSentEvent

from beeai_framework.adapters.openai.serve._permissions import (
    APPROVAL_FUNCTION,
    ApprovalRequest,
    OpenAIRun,
)
from beeai_framework.adapters.openai.serve.responses import _types as types
from beeai_framework.backend import AnyMessage
from beeai_framework.utils.strings import to_json


def approval_output(approval: ApprovalRequest) -> types.ResponsesFunctionCallOutput:
    return types.ResponsesFunctionCallOutput(
        id=f"fc_{uuid.uuid4().hex}",
        call_id=approval.call_id,
        name=APPROVAL_FUNCTION,
        arguments=approval.arguments,
    )


async def stream_response(
    run: OpenAIRun,
    response_id: str,
    to_message: Callable[[types.ResponsesResponseOutput], AnyMessage],
) -> AsyncIterator[ServerSentEvent]:
    sequence = 0
    outputs: list[types.ResponsesResponseOutput] = []
    current: types.ResponsesMessageOutput | None = None
    text = ""
    suspended = False

    def event(kind: str, **data: Any) -> ServerSentEvent:
        nonlocal sequence
        result = ServerSentEvent(data=to_json({"type": kind, "sequence_number": sequence, **data}), event=kind)
        sequence += 1
        return result

    def response(status: str) -> dict[str, Any]:
        return types.ResponsesResponse(
            id=response_id, created=int(time.time()), status=status, model=run.model_id, output=outputs.copy()
        ).model_dump()

    def finish_message() -> list[ServerSentEvent]:
        nonlocal current, text
        if current is None:
            return []
        index = len(outputs)
        current.status = "completed"
        current.content = [types.ResponsesMessageContent(text=text)]
        result = [
            event("response.output_text.done", output_index=index, item_id=current.id, content_index=0, text=text),
            event(
                "response.content_part.done",
                output_index=index,
                item_id=current.id,
                content_index=0,
                part=current.content[0].model_dump(),
            ),
            event("response.output_item.done", output_index=index, item=current.model_dump()),
        ]
        outputs.append(current)
        run.memory_messages.append(to_message(current))
        current, text = None, ""
        return result

    try:
        yield event("response.created", response=response("in_progress"))
        yield event("response.in_progress", response=response("in_progress"))
        async with aclosing(run.events()) as events:
            async for message in events:
                if isinstance(message, ApprovalRequest) or message.type != "message" or not message.append:
                    for item in finish_message():
                        yield item
                if isinstance(message, ApprovalRequest):
                    approval = approval_output(message)
                    index = len(outputs)
                    yield event("response.output_item.added", output_index=index, item=approval.model_dump())
                    yield event(
                        "response.function_call_arguments.done",
                        output_index=index,
                        item_id=approval.id,
                        arguments=approval.arguments,
                        name=approval.name,
                    )
                    yield event("response.output_item.done", output_index=index, item=approval.model_dump())
                    outputs.append(approval)
                    suspended = True
                elif message.type == "message":
                    if current is None:
                        current = types.ResponsesMessageOutput(id=f"msg_{uuid.uuid4().hex}")
                        yield event("response.output_item.added", output_index=len(outputs), item=current.model_dump())
                        yield event(
                            "response.content_part.added",
                            output_index=len(outputs),
                            item_id=current.id,
                            content_index=0,
                            part=types.ResponsesMessageContent(text="").model_dump(),
                        )
                    text += message.text
                    yield event(
                        "response.output_text.delta",
                        output_index=len(outputs),
                        item_id=current.id,
                        content_index=0,
                        delta=message.text,
                    )
                else:
                    output: types.ResponsesResponseOutput
                    if message.type == "reasoning":
                        output = types.ResponsesReasoningOutput(
                            id=f"rs_{uuid.uuid4().hex}",
                            status="completed",
                            content=types.ResponsesReasoningContent(text=message.text),
                        )
                    else:
                        output = types.ResponsesCustomToolCallOutput(
                            id=f"ctc_{uuid.uuid4().hex}",
                            name="tools_call",
                            input=message.text,
                            call_id=str(uuid.uuid4()),
                        )
                    yield event("response.output_item.added", output_index=len(outputs), item=output.model_dump())
                    yield event("response.output_item.done", output_index=len(outputs), item=output.model_dump())
                    outputs.append(output)
                    run.memory_messages.append(to_message(output))
        for item in finish_message():
            yield item
        if not suspended and run.memory is not None:
            await run.memory.add_many(run.memory_messages)
        yield event("response.completed", response=response("completed"))
    except Exception as exc:
        yield event("error", code="server_error", message=str(exc), param=None)
        yield event("response.failed", response=response("failed"))
    finally:
        # A stream that was not fully delivered cannot leave a waiting invocation behind.
        if not suspended:
            await run.close()
