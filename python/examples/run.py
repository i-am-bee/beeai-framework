import os
import pathlib
import runpy
import sys

from dotenv import load_dotenv

load_dotenv()

EXAMPLES_DIR = (pathlib.Path(__file__).parent).resolve()
all_examples = list(EXAMPLES_DIR.rglob("*.py"))

exclude = list(
    filter(
        None,
        [
            "_*.py",
            "helpers/io.py",
        ],
    )
)


def example_name(path: pathlib.Path) -> str:
    return str(path.relative_to(EXAMPLES_DIR)).replace(os.sep, "/")


def is_excluded(path: pathlib.Path) -> bool:
    for pattern in exclude:
        if "/**" in pattern:
            raise ValueError("Double star '**' is not supported!")

        if path.match(pattern):
            return True
    return False


examples = sorted(
    {example for example in all_examples if not is_excluded(example)},
    key=example_name,
)


if __name__ == "__main__":
    example = pathlib.Path(sys.argv[1]).resolve()
    if example in examples:
        runpy.run_path(str(example.resolve()), run_name="__main__")
    else:
        print(f"Not a valid example script: {example}")
