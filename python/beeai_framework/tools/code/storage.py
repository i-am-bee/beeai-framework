# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel

from beeai_framework.tools.errors import ToolError


class PythonFile(BaseModel):
    id: str
    python_id: str
    filename: str


class PythonStorage(ABC):
    """
    Abstract class for managing files in Python code interpreter.
    """

    @abstractmethod
    async def list_files(self) -> list[PythonFile]:
        """
        List all files that code interpreter can use.
        """
        pass

    @abstractmethod
    async def upload(self, files: list[PythonFile]) -> list[PythonFile]:
        """
        Prepare subset of available files to code interpreter.
        """
        pass

    @abstractmethod
    async def download(self, files: list[PythonFile]) -> list[PythonFile]:
        """
        Process updated/modified/deleted files from code interpreter response.
        """
        pass


class LocalPythonStorage(PythonStorage):
    def __init__(
        self, *, local_working_dir: str, interpreter_working_dir: str, ignored_files: list[str] | None = None
    ) -> None:
        self._local_working_dir = local_working_dir
        self._interpreter_working_dir = interpreter_working_dir
        self._ignored_files = ignored_files or []

    @property
    def local_working_dir(self) -> str:
        return self._local_working_dir

    @property
    def interpreter_working_dir(self) -> str:
        return self._interpreter_working_dir

    @property
    def ignored_files(self) -> list[str]:
        return self._ignored_files

    def init(self) -> None:
        os.makedirs(self._local_working_dir, exist_ok=True)
        os.makedirs(self._interpreter_working_dir, exist_ok=True)

    async def list_files(self) -> list[PythonFile]:
        self.init()
        files = os.listdir(self._local_working_dir)
        python_files = []
        for file in files:
            python_id = self._compute_hash(os.path.join(self._local_working_dir, file))
            python_files.append(
                PythonFile(
                    filename=file,
                    id=python_id,
                    python_id=python_id,
                )
            )
        return python_files

    async def upload(self, files: list[PythonFile]) -> list[PythonFile]:
        self.init()

        for file in files:
            source_path = self._resolve_safe_path(self._local_working_dir, file.filename)
            shutil.copyfile(
                str(source_path),
                os.path.join(self._interpreter_working_dir, file.python_id),
            )
        return files

    async def download(self, files: list[PythonFile]) -> list[PythonFile]:
        self.init()

        for file in files:
            target_path = self._resolve_safe_path(self._local_working_dir, file.filename)
            os.makedirs(target_path.parent, exist_ok=True)
            shutil.copyfile(
                os.path.join(self._interpreter_working_dir, file.python_id),
                str(target_path),
            )
        return files

    @staticmethod
    def _resolve_safe_path(base_dir: str, filename: str) -> Path:
        base_path = Path(base_dir).resolve()
        target_path = base_path.joinpath(filename).resolve()
        if not target_path.is_relative_to(base_path):
            raise ToolError(f"Path traversal detected in filename: {filename}")
        return target_path

    @staticmethod
    def _compute_hash(file_path: str) -> str:
        with open(file_path, "rb") as f:
            digest = hashlib.file_digest(f, "sha256")
            return digest.hexdigest()
