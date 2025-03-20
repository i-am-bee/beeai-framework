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

import hashlib
import os
import shutil
from abc import ABC, abstractmethod

from pydantic import BaseModel


class PythonFile(BaseModel):
    id: str
    python_id: str
    filename: str


class PythonStorage(ABC):
    """
    Abstract class for managing files in Python code interpreter.
    """

    @abstractmethod
    def list_files(self) -> list[PythonFile]:
        """
        List all files that code interpreter can use.
        """
        pass

    @abstractmethod
    def upload(self, files: list[PythonFile]) -> list[PythonFile]:
        """
        Prepare subset of available files to code interpreter.
        """
        pass

    @abstractmethod
    def download(self, files: list[PythonFile]) -> list[PythonFile]:
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

    def list_files(self) -> list[PythonFile]:
        self.init()
        files = os.listdir(self._local_working_dir)
        python_files = []
        for file in files:
            python_id = self._compute_hash(self._local_working_dir + "/" + file)
            python_files.append(
                PythonFile(
                    filename=file,
                    id=python_id,
                    python_id=python_id,
                )
            )
        return python_files

    def upload(self, files: list[PythonFile]) -> list[PythonFile]:
        self.init()

        for file in files:
            shutil.copyfile(
                os.path.join(self._local_working_dir, file.filename),
                os.path.join(self._interpreter_working_dir, file.python_id),
            )
        return files

    def download(self, files: list[PythonFile]) -> list[PythonFile]:
        self.init()

        for file in files:
            shutil.copyfile(
                os.path.join(self._interpreter_working_dir, file.python_id),
                os.path.join(self._local_working_dir, file.filename),
            )
        return files

    def clean_up(self, rmtree: bool = False) -> None:
        if rmtree:
            shutil.rmtree(self._local_working_dir)
            shutil.rmtree(self._interpreter_working_dir)
        else:
            os.rmdir(self._local_working_dir)
            os.rmdir(self._interpreter_working_dir)

    @staticmethod
    def _compute_hash(file_path: str) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
            return hasher.hexdigest()
