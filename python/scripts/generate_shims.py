import os.path
import textwrap
from pathlib import Path

MAPPINGS = {
    "agents/experimental/__init__.py": "agents/requirement/__init__.py",
    "agents/experimental/_utils.py": "agents/requirement/utils/__init__.py",
    "agents/experimental/agent.py": "agents/requirement/agent.py",
    "agents/experimental/events.py": "agents/requirement/events.py",
    "agents/experimental/prompts.py": "agents/requirement/prompts.py",
    "agents/experimental/types.py": "agents/requirement/types.py",
    "agents/experimental/utils/__init__.py": "agents/requirement/utils/__init__.py",
    "agents/experimental/utils/_llm.py": "agents/requirement/utils/_llm.py",
    "agents/experimental/utils/_tool.py": "agents/requirement/utils/_tool.py",
    "agents/experimental/requirements/__init__.py": "agents/requirement/requirements/__init__.py",
    "agents/experimental/requirements/_utils.py": "agents/requirement/requirements/_utils.py",
    "agents/experimental/requirements/ask_permission.py": "agents/requirement/requirements/ask_permission.py",
    "agents/experimental/requirements/conditional.py": "agents/requirement/requirements/conditional.py",
    "agents/experimental/requirements/events.py": "agents/requirement/requirements/events.py",
    "agents/experimental/requirements/requirement.py": "agents/requirement/requirements/requirement.py",
}


def to_import_path(path: str) -> str:
    base, ext = os.path.splitext(path)
    if os.path.basename(base) == "__init__":
        base = os.path.dirname(base)

    # Normalize filesystem separators into dots
    dotted = base.replace(os.sep, ".").replace("\\", ".")

    # Handle leading './' or similar
    dotted = dotted.lstrip(".")

    return dotted


def make_shim(old: str, new: str, path: Path) -> None:
    code = textwrap.dedent(
        f"""\
        import sys
        import warnings
        import {new} as _new_module

        warnings.warn(
            "{old} is deprecated and will be removed in a future release. "
            "Please use {new} instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        sys.modules[__name__] = _new_module
        """
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code)


def main() -> None:
    root = Path(__file__).parent.parent

    for _old_path, _new_path in MAPPINGS.items():
        # Check for a presence
        new = to_import_path(f"beeai_framework/{_new_path}")
        new_path = Path(root, "beeai_framework", _new_path)

        if not new_path.exists():
            raise FileNotFoundError(f"File {new_path} does not exist")

        old_path = Path(root, "beeai_framework", _old_path)
        old = to_import_path(f"beeai_framework/{_old_path}")

        # Clean any existing shim
        if old_path.exists():
            for p in old_path.rglob("__init__.py"):
                p.unlink()

        # Generate new shim
        print(f"Generating shim: {old} -> {new}")
        make_shim(old, new, old_path)


if __name__ == "__main__":
    main()
