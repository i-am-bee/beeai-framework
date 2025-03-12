# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterable
from typing import Any

from commitizen import git
from commitizen.cz.conventional_commits import ConventionalCommitsCz  # type: ignore

__all__ = ["MonorepoCommitsCz"]

from commitizen.defaults import Questions


class MonorepoCommitsCz(ConventionalCommitsCz):
    change_type_map = {  # noqa: RUF012
        "feat": "Features",
        "fix": "Bug Fixes",
        "refactor": "Refactor",
        "perf": "Performance Improvements",
    }

    def changelog_message_builder_hook(
        self, parsed_message: dict[str, Any], commit: git.GitCommit
    ) -> dict[str, Any] | Iterable[dict[str, Any]] | None:
        changed_files = git.get_filenames_in_commit(commit.rev) or []

        has_python_changes = any(file.startswith("python/") for file in changed_files)
        if not has_python_changes:
            return None

        parent_hook = super().changelog_message_builder_hook
        return parent_hook(parsed_message, commit) if parent_hook else parsed_message

    def questions(self) -> Questions:
        questions = super().questions()
        for index, question in enumerate(questions):
            if question["name"] == "prefix":
                question["choices"].append({"value": "chore", "name": "chore: other uncategorized changes"})

            if question["name"] == "scope":
                questions[index] = {  # type: ignore
                    "type": "list",
                    "name": "scope",
                    "message": "What is the scope of this change?",
                    "filter": lambda value: value or "",
                    "choices": [
                        {"name": name or "", "value": name}
                        for name in [
                            None,
                            "adapters",
                            "agents",
                            "backend",
                            "tools",
                            "cache",
                            "emitter",
                            "examples",
                            "internals",
                            "logger",
                            "memory",
                            "serializer",
                            "infra",
                            "deps",
                            "instrumentation",
                            "workflows",
                        ]
                    ],
                }
                break

        return questions
