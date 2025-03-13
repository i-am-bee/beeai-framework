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
        self.localWorkingDir = local_working_dir
        self.interpreterWorkingDir = interpreter_working_dir
        self.ignoredFiles = ignored_files


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
        self.input = {**input, "ignoredFiles": []}

    def init(self) -> None:
        # await os.makedirs(self.input["localWorkingDir"], exist_ok=True)
        # await os.makedirs(self.input["interpreterWorkingDir"], exist_ok=True)
        os.makedirs(self.input.get("localWorkingDir"), exist_ok=True)
        os.makedirs(self.input.get("interpreterWorkingDir"), exist_ok=True)

    def list(self) -> [PythonFile]:
        self.init()
        files = os.listdir(self.input.get("localWorkingDir"))
        return [
            {
                "id": compute_hash(self.input.get("localWorkingDir") + "/" + file),
                "filename": file,
                "pythonId": compute_hash(self.input.get("localWorkingDir") + "/" + file),
            }
            for file in files
        ]

    async def upload(self, files: PythonUploadFile) -> PythonUploadFile:
        self.init()

        for file in files:
            shutil.copyfile(
                os.path.join(self.input.get("localWorkingDir"), file["filename"]),
                os.path.join(self.input.get("interpreterWorkingDir"), file["pythonId"]),
            )
        return files

    async def download(self, files: PythonDownloadFile) -> PythonDownloadFile:
        self.init()

        for file in files:
            shutil.copyfile(
                os.path.join(self.input.get("interpreterWorkingDir"), file["pythonId"]),
                os.path.join(self.input.get("localWorkingDir"), file["filename"]),
            )
        return files


def compute_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
        return hasher.hexdigest()
