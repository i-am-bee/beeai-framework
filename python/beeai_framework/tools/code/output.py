# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from beeai_framework.tools import ToolOutput
from beeai_framework.tools.code.storage import PythonFile


class PythonToolOutput(ToolOutput):
    FILE_PREFIX = "urn:bee:file"

    def __init__(self, stdout: str, stderr: str, exit_code: int, output_files: list[PythonFile]) -> None:
        super().__init__()

        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.output_files = output_files

    def is_empty(self) -> bool:
        return False

    def get_text_content(self) -> str:
        execution_status = (
            "The code executed successfully."
            if self.exit_code == 0
            else f"The code exited with error code {self.exit_code}."
        )
        stdout = f"Standard output:\n\n```\n{self.stdout}\n```" if self.stdout and self.stdout.strip() else None
        stderr = f"Error output:\n\n```\n{self.stderr}\n```" if self.stderr and self.stderr.strip() else None

        def is_image(filename: str) -> bool:
            return any(filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp"])

        files = (
            "The following files were created or modified. The user does not see them yet. \
            To present a file to the user, send them the link below, verbatim:\n"
            + "\n".join(
                f"{'!' if is_image(file.filename) else ''}[ {file.filename} ]\
                ({PythonToolOutput.FILE_PREFIX}:{file.id})"
                for file in self.output_files
            )
            if self.output_files
            else None
        )
        message = ""
        for x in [execution_status, stdout, stderr, files]:
            if bool(x):
                message = message + str(x) + "\n"
        return message
