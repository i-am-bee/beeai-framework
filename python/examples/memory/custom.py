from typing import Any

from beeai_framework.backend.message import AnyMessage
from beeai_framework.memory import BaseMemory


class MyMemory(BaseMemory):
    @property
    def messages(self) -> list[AnyMessage]:
        raise NotImplementedError("Method not yet implemented.")

    async def add(self, message: AnyMessage, index: int | None = None) -> None:
        raise NotImplementedError("Method not yet implemented.")

    async def delete(self, message: AnyMessage) -> bool:
        raise NotImplementedError("Method not yet implemented.")

    def reset(self) -> None:
        raise NotImplementedError("Method not yet implemented.")

    def create_snapshot(self) -> Any:
        raise NotImplementedError("Method not yet implemented.")

    def load_snapshot(self, state: Any) -> None:
        raise NotImplementedError("Method not yet implemented.")
