from pydantic import BaseModel
from termcolor import colored

from beeai_framework.utils.models import ModelLike, to_model_optional


class ReaderOptions(BaseModel):
    fallback: str = ""
    input: str = "User 👤 : "
    allow_empty: bool = False


class ConsoleReader:
    def __init__(self, options: ModelLike[ReaderOptions] | None = None) -> None:
        options = to_model_optional(ReaderOptions, options) or ReaderOptions()
        self.fallback = options.fallback
        self.input = options.input
        self.allow_empty = options.allow_empty

    def __iter__(self) -> "ConsoleReader":
        print("Interactive session has started. To escape, input 'q' and submit.")
        return self

    def __next__(self) -> str:
        try:
            while True:
                prompt = input(colored(self.input, "cyan", attrs=["bold"])).strip()

                if prompt == "q":
                    raise StopIteration

                prompt = prompt if prompt else self.fallback

                if not prompt and not self.allow_empty:
                    print("Error: Empty prompt is not allowed. Please try again.")
                    continue

                return prompt
        except (EOFError, KeyboardInterrupt):
            print()
            exit()

    def write(self, role: str, data: str) -> None:
        print(colored(role, "red", attrs=["bold"]), data)

    def prompt(self) -> str | None:
        for prompt in self:
            return prompt
        exit()

    def ask_single_question(self, query_message: str) -> str:
        answer = input(colored(query_message, "cyan", attrs=["bold"]))
        return answer.strip()
