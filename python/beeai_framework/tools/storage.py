import hashlib
import os
import shutil
from abc import abstractmethod


class PythonFile:
    def __init__(self, filename: str, id: str, python_id: str) -> None:
        self.id = id
        self.pythonId = python_id
        self.filename = filename


class PythonUploadFile:
    def __init__(self, filename: str, id: str) -> None:
        self.id = id
        self.filename = filename


class PythonDownloadFile:
    def __init__(self, filename: str, id: str, python_id: str) -> None:
        self.id = id
        self.filename = filename
        self.pythonId = python_id


class Input:
    def __init__(self, local_working_dir: str, interpreter_working_dir: str, ignored_files: [str]) -> None:
        self._local_working_dir = local_working_dir
        self._interpreter_working_dir = interpreter_working_dir
        self._ignored_files = ignored_files

    @property
    def local_working_dir(self) -> str:
        return self._local_working_dir

    @property
    def interpreter_working_dir(self) -> str:
        return self._interpreter_working_dir

    @property
    def ignored_files(self) -> [str]:
        return self._ignored_files

    @local_working_dir.setter
    def local_working_dir(self, v: str) -> None:
        self._local_working_dir = v

    @interpreter_working_dir.setter
    def interpreter_working_dir(self, v:str) -> None:
        self._interpreter_working_eir = v

    @ignored_files.setter
    def ignored_files(self, v: [str]) -> None:
        self._ignored_files = v

class PythonStorage:
    """
    Abstract class for managing files in Python code interpreter.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def list(self) -> [PythonFile]:
        """
        List all files that code interpreter can use.
        """
        pass

    @abstractmethod
    def upload(self, files: PythonUploadFile) -> PythonUploadFile:
        """
        Prepare subset of available files to code interpreter.
        """
        pass

    @abstractmethod
    def download(self, files: PythonDownloadFile) -> PythonDownloadFile:
        """
        Process updated/modified/deleted files from code interpreter response.
        """
        pass


class LocalPythonStorage(PythonStorage):
    def __init__(self, input: Input) -> None:
        super().__init__()
        self.input = input

    def init(self) -> None:
        # await os.makedirs(self.input["localWorkingDir"], exist_ok=True)
        # await os.makedirs(self.input["interpreterWorkingDir"], exist_ok=True)
        os.makedirs(self.input.local_working_dir, exist_ok=True)
        os.makedirs(self.input.interpreter_working_dir, exist_ok=True)

    def list(self) -> [PythonFile]:
        self.init()
        files = os.listdir(self.input.local_working_dir)
        return [
            {
                "id": compute_hash(self.input.local_working_dir + "/" + file),
                "filename": file,
                "pythonId": compute_hash(self.input.local_working_dir + "/" + file),
            }
            for file in files
        ]

    async def upload(self, files: PythonUploadFile) -> PythonUploadFile:
        self.init()

        for file in files:
            shutil.copyfile(
                os.path.join(self.input.local_working_dir, file["filename"]),
                os.path.join(self.input.interpreter_working_dir, file["pythonId"]),
            )
        return files

    async def download(self, files: PythonDownloadFile) -> PythonDownloadFile:
        self.init()

        for file in files:
            shutil.copyfile(
                os.path.join(self.input.interpreter_working_dir, file["pythonId"]),
                os.path.join(self.input.local_working_dir, file["filename"]),
            )
        return files


def compute_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
        return hasher.hexdigest()
