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
            source_path = self._resolve_safe_path(self._local_working_dir, file.filename, follow_symlinks=False)
            dest_path = self._resolve_safe_path(self._interpreter_working_dir, file.python_id, follow_symlinks=False)
            shutil.copyfile(str(source_path), str(dest_path))
        return files

    async def download(self, files: list[PythonFile]) -> list[PythonFile]:
        self.init()

        for file in files:
            source_path = self._resolve_safe_path(self._interpreter_working_dir, file.python_id)
            target_path = self._resolve_safe_path(self._local_working_dir, file.filename)
            os.makedirs(target_path.parent, exist_ok=True)
            shutil.copyfile(str(source_path), str(target_path))
        return files

    @staticmethod
    def _resolve_safe_path(base_dir: str, name: str, *, follow_symlinks: bool = True) -> Path:
        """Resolve a file path and ensure it stays within the base directory.

        Args:
            base_dir: The directory the path must stay within.
            name: The untrusted filename or relative path to validate.
            follow_symlinks: If True, resolve symlinks (use for untrusted names).
                If False, use lexical normalization only (use for trusted names
                where symlinks should be preserved).

        Returns:
            The validated absolute path.
        """
        base = Path(base_dir).resolve()
        target = base / name
        target = target.resolve() if follow_symlinks else Path(os.path.normpath(target))
        if not target.is_relative_to(base):
            raise ToolError(f"Path traversal detected: {name}")
        return target

    @staticmethod
    def _compute_hash(file_path: str) -> str:
        with open(file_path, "rb") as f:
            digest = hashlib.file_digest(f, "sha256")
            return digest.hexdigest()
