# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import uuid
from collections.abc import Generator
from typing import Any

import pytest
import pytest_asyncio

from beeai_framework.tools.code.storage import LocalPythonStorage, PythonFile
from beeai_framework.tools.errors import ToolError


def _write_file(path: str, content: str) -> None:
    with open(path, "w") as f:
        f.write(content)


def _read_file(path: str) -> str:
    with open(path) as f:
        return f.read()


def _file_exists(path: str) -> bool:
    return os.path.exists(path)


@pytest_asyncio.fixture
def storage_dirs() -> Generator[tuple[str, str], Any, None]:
    """Create temporary local and interpreter directories for testing."""
    dir_id = str(uuid.uuid4())
    local_dir = f"/tmp/test_storage_local_{dir_id}"
    interpreter_dir = f"/tmp/test_storage_interpreter_{dir_id}"
    os.makedirs(local_dir, exist_ok=True)
    os.makedirs(interpreter_dir, exist_ok=True)

    yield local_dir, interpreter_dir

    shutil.rmtree(local_dir, ignore_errors=True)
    shutil.rmtree(interpreter_dir, ignore_errors=True)


@pytest_asyncio.fixture
def storage(storage_dirs: tuple[str, str]) -> LocalPythonStorage:
    local_dir, interpreter_dir = storage_dirs
    return LocalPythonStorage(local_working_dir=local_dir, interpreter_working_dir=interpreter_dir)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_download_path_traversal_blocked(storage: LocalPythonStorage, storage_dirs: tuple[str, str]) -> None:
    """A filename with ../ components must be rejected by download()."""
    _, interpreter_dir = storage_dirs

    fake_id = "malicious_hash_id"
    _write_file(os.path.join(interpreter_dir, fake_id), "malicious payload")

    malicious_file = PythonFile(
        id=fake_id,
        python_id=fake_id,
        filename="../../../../tmp/pwned_by_traversal.txt",
    )

    with pytest.raises(ToolError, match="Path traversal detected in filename"):
        await storage.download([malicious_file])

    assert not _file_exists("/tmp/pwned_by_traversal.txt")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upload_path_traversal_blocked(storage: LocalPythonStorage, storage_dirs: tuple[str, str]) -> None:
    """A filename with ../ components must be rejected by upload()."""
    malicious_file = PythonFile(
        id="some_id",
        python_id="some_id",
        filename="../../../etc/passwd",
    )

    with pytest.raises(ToolError, match="Path traversal detected in filename"):
        await storage.upload([malicious_file])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_download_safe_filename_works(storage: LocalPythonStorage, storage_dirs: tuple[str, str]) -> None:
    """A normal filename without traversal should work fine."""
    local_dir, interpreter_dir = storage_dirs

    file_id = "safe_hash_id"
    _write_file(os.path.join(interpreter_dir, file_id), "safe content")

    safe_file = PythonFile(id=file_id, python_id=file_id, filename="output.txt")

    result = await storage.download([safe_file])
    assert len(result) == 1

    target = os.path.join(local_dir, "output.txt")
    assert _file_exists(target)
    assert _read_file(target) == "safe content"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_download_subdirectory_filename_works(storage: LocalPythonStorage, storage_dirs: tuple[str, str]) -> None:
    """A filename with a subdirectory (no traversal) should work fine."""
    local_dir, interpreter_dir = storage_dirs

    file_id = "subdir_hash_id"
    _write_file(os.path.join(interpreter_dir, file_id), "subdir content")

    safe_file = PythonFile(id=file_id, python_id=file_id, filename="subdir/output.txt")

    result = await storage.download([safe_file])
    assert len(result) == 1

    target = os.path.join(local_dir, "subdir", "output.txt")
    assert _file_exists(target)
    assert _read_file(target) == "subdir content"
