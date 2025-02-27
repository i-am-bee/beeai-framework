from typing import Any

from beeai_framework.backend.message import Message
from beeai_framework.memory import BaseMemory


class MyMemory(BaseMemory):
    @property
    def messages(self) -> list[Message]:
        raise NotImplementedError("Method not yet implemented.")

    def add(self, message: Message, index: int | None = None) -> None:
        raise NotImplementedError("Method not yet implemented.")

    def delete(self, message: Message) -> bool:
        raise NotImplementedError("Method not yet implemented.")

    def reset(self) -> None:
        raise NotImplementedError("Method not yet implemented.")

    def create_snapshot(self) -> Any:
        raise NotImplementedError("Method not yet implemented.")

    def load_snapshot(self, state: Any) -> None:
        raise NotImplementedError("Method not yet implemented.")
