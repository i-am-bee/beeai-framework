# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest

from beeai_framework.tools.code.storage import LocalPythonStorage, PythonFile
from beeai_framework.tools.errors import ToolError


@pytest.fixture
def storage(tmp_path: Path) -> LocalPythonStorage:
    """Create a LocalPythonStorage backed by pytest tmp_path."""
    local_dir = tmp_path / "local"
    interpreter_dir = tmp_path / "interpreter"
    local_dir.mkdir()
    interpreter_dir.mkdir()
    return LocalPythonStorage(
        local_working_dir=str(local_dir),
        interpreter_working_dir=str(interpreter_dir),
    )


# ── download: filename traversal ────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
async def test_download_filename_traversal_blocked(storage: LocalPythonStorage) -> None:
    """download() must reject a filename with ../ components."""
    interp = Path(storage.interpreter_working_dir)
    (interp / "fake_id").write_text("payload")

    bad_file = PythonFile(id="fake_id", python_id="fake_id", filename="../../../../tmp/pwned.txt")

    with pytest.raises(ToolError, match="Path traversal detected"):
        await storage.download([bad_file])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_download_safe_filename_works(storage: LocalPythonStorage) -> None:
    """A normal filename without traversal should succeed."""
    interp = Path(storage.interpreter_working_dir)
    (interp / "hash_id").write_text("safe content")

    safe_file = PythonFile(id="hash_id", python_id="hash_id", filename="output.txt")
    result = await storage.download([safe_file])

    assert len(result) == 1
    target = Path(storage.local_working_dir) / "output.txt"
    assert target.read_text() == "safe content"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_download_subdirectory_filename_works(storage: LocalPythonStorage) -> None:
    """A filename with a subdirectory (no traversal) should succeed."""
    interp = Path(storage.interpreter_working_dir)
    (interp / "sub_id").write_text("sub content")

    safe_file = PythonFile(id="sub_id", python_id="sub_id", filename="subdir/output.txt")
    result = await storage.download([safe_file])

    assert len(result) == 1
    target = Path(storage.local_working_dir) / "subdir" / "output.txt"
    assert target.read_text() == "sub content"


# ── download: python_id traversal ───────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
async def test_download_python_id_absolute_path_blocked(storage: LocalPythonStorage) -> None:
    """download() must reject a python_id that is an absolute path."""
    bad_file = PythonFile(id="x", python_id="/etc/passwd", filename="stolen.txt")

    with pytest.raises(ToolError, match="Path traversal detected"):
        await storage.download([bad_file])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_download_python_id_relative_traversal_blocked(storage: LocalPythonStorage) -> None:
    """download() must reject a python_id with ../ components."""
    bad_file = PythonFile(id="x", python_id="../../etc/shadow", filename="stolen.txt")

    with pytest.raises(ToolError, match="Path traversal detected"):
        await storage.download([bad_file])


# ── upload: filename traversal ──────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upload_filename_traversal_blocked(storage: LocalPythonStorage) -> None:
    """upload() must reject a filename with ../ components."""
    bad_file = PythonFile(id="x", python_id="x", filename="../../../etc/passwd")

    with pytest.raises(ToolError, match="Path traversal detected"):
        await storage.upload([bad_file])


# ── upload: python_id traversal ─────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upload_python_id_absolute_path_blocked(storage: LocalPythonStorage) -> None:
    """upload() must reject a python_id that is an absolute path."""
    local = Path(storage.local_working_dir)
    (local / "legit.txt").write_text("data")

    bad_file = PythonFile(id="x", python_id="/etc/cron.d/backdoor", filename="legit.txt")

    with pytest.raises(ToolError, match="Path traversal detected"):
        await storage.upload([bad_file])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upload_python_id_relative_traversal_blocked(storage: LocalPythonStorage) -> None:
    """upload() must reject a python_id with ../ components."""
    local = Path(storage.local_working_dir)
    (local / "legit.txt").write_text("data")

    bad_file = PythonFile(id="x", python_id="../../tmp/evil", filename="legit.txt")

    with pytest.raises(ToolError, match="Path traversal detected"):
        await storage.upload([bad_file])


# ── upload: symlink preservation ─────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upload_symlinked_file_works(storage: LocalPythonStorage, tmp_path: Path) -> None:
    """upload() must handle symlinked input files without rejecting them.

    Upload filenames come from os.listdir(local_working_dir) and are trusted.
    Using normpath (not resolve) ensures symlinks pointing outside the dir work.
    """
    real_file = tmp_path / "real_data.csv"
    real_file.write_text("csv,data")

    local = Path(storage.local_working_dir)
    link_path = local / "data.csv"
    os.symlink(str(real_file), str(link_path))

    interp = Path(storage.interpreter_working_dir)
    sym_file = PythonFile(id="link_hash", python_id="link_hash", filename="data.csv")

    result = await storage.upload([sym_file])

    assert len(result) == 1
    dest = interp / "link_hash"
    assert dest.read_text() == "csv,data"
