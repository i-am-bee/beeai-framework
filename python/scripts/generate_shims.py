# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

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


SHIM_FOOTER = "sys.modules[__name__] = _new_module"


def create_shim(*, old_module: str, new_module: str) -> str:
    return textwrap.dedent(
        f"""\
        import sys
        import warnings
        import {new_module} as _new_module

        warnings.warn(
            "{old_module} is deprecated and will be removed in a future release. "
            "Please use {new_module} instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        {SHIM_FOOTER}
        """
    )


def main() -> None:
    root = Path(__file__).parent.parent
    project_name = "beeai_framework"

    for _old_path, _new_path in MAPPINGS.items():
        new = to_import_path(f"{project_name}/{_new_path}")
        new_path = Path(root, project_name, _new_path)
        if not new_path.exists():
            raise FileNotFoundError(f"File {new_path} does not exist")

        old_path = Path(root, project_name, _old_path)
        old = to_import_path(f"{project_name}/{_old_path}")

        if old_path.exists():
            content = old_path.read_text()
            if SHIM_FOOTER in content:
                print(f"Skipping shim: {old} -> {new}")
                continue

        print(f"Generating shim: {old} -> {new}")
        code = create_shim(old_module=old, new_module=new)
        old_path.parent.mkdir(parents=True, exist_ok=True)
        old_path.write_text(code)


if __name__ == "__main__":
    main()
