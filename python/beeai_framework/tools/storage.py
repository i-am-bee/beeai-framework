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
from abc import abstractmethod


class PythonFile:
    def __init__(self, filename: str, id: str, python_id: str) -> None:
        self._id = id
        self._python_id = python_id
        self._filename = filename

    @property
    def id(self) -> str:
        return self._id

    @id.setter
    def id(self, value: str) -> None:
        self._id = value

    @property
    def python_id(self) -> str:
        return self._python_id

    @python_id.setter
    def python_id(self, value: str) -> None:
        self._python_id = value

    @property
    def filename(self) -> str:
        return self._filename

    @filename.setter
    def filename(self, value: str) -> None:
        self._filename = value


class Input:
    def __init__(self, local_working_dir: str, interpreter_working_dir: str, ignored_files: list[str]) -> None:
        self._local_working_dir = local_working_dir
        self._interpreter_working_dir = interpreter_working_dir
        self._ignored_files = ignored_files

    @property
    def local_working_dir(self) -> str:
        return self._local_working_dir

    @local_working_dir.setter
    def local_working_dir(self, v: str) -> None:
        self._local_working_dir = v

    @property
    def interpreter_working_dir(self) -> str:
        return self._interpreter_working_dir

    @interpreter_working_dir.setter
    def interpreter_working_dir(self, v: str) -> None:
        self._interpreter_working_eir = v

    @property
    def ignored_files(self) -> list[str]:
        return self._ignored_files

    @ignored_files.setter
    def ignored_files(self, v: list[str]) -> None:
        self._ignored_files = v


class PythonStorage:
    """
    Abstract class for managing files in Python code interpreter.
    """

    def __init__(self) -> None:
        super().__init__()

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
    def __init__(self, input: Input) -> None:
        super().__init__()
        self.input = input

    def init(self) -> None:
        os.makedirs(self.input.local_working_dir, exist_ok=True)
        os.makedirs(self.input.interpreter_working_dir, exist_ok=True)

    def list_files(self) -> list[PythonFile]:
        self.init()
        files = os.listdir(self.input.local_working_dir)
        return [
            PythonFile(
                file,
                compute_hash(self.input.local_working_dir + "/" + file),
                compute_hash(self.input.local_working_dir + "/" + file),
            )
            for file in files
        ]

    def upload(self, files: list[PythonFile]) -> list[PythonFile]:
        self.init()

        for file in files:
            shutil.copyfile(
                os.path.join(self.input.local_working_dir, file.filename),
                os.path.join(self.input.interpreter_working_dir, file.python_id),
            )
        return files

    def download(self, files: list[PythonFile]) -> list[PythonFile]:
        self.init()

        for file in files:
            shutil.copyfile(
                os.path.join(self.input.interpreter_working_dir, file.python_id),
                os.path.join(self.input.local_working_dir, file.filename),
            )
        return files


def compute_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
        return hasher.hexdigest()
