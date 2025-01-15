import os

from dotenv import load_dotenv
from github import Github
from github.Repository import Repository

from logger import get_logger


REPO_OWNER = "giamberinigiulia"
REPO_NAME = "SPE_Project"
DATASET_PATH = "data/python_files.txt"

unique_files = set()
logger = get_logger(__name__)


def collect_dataset():
    repo = _get_github_repo()
    python_files = _traverse_contents(repo, "")
    contents = [content for _, content in python_files]
    _write_to_file(DATASET_PATH, contents)


def _get_github_repo() -> Repository:
    g = Github(_get_github_token())
    return g.get_repo(f"{REPO_OWNER}/{REPO_NAME}")


def _get_github_token() -> str:
    load_dotenv()
    return os.environ["GITHUB_TOKEN"]


def _traverse_contents(repo: Repository, path: str) -> list[tuple[str, str]]:
    python_files = []
    contents = repo.get_contents(path)

    if not isinstance(contents, list):
        contents = [contents]

    for file in contents:
        if file.type == "file" and file.name.endswith(".py"):
            if file.path not in unique_files:
                unique_files.add(file.path)
                python_files.append(_get_file_content(repo, file.path))
        elif file.type == "dir":
            python_files.extend(_traverse_contents(repo, file.path))

    return python_files


def _get_file_content(repo: Repository, path: str) -> tuple[str, str]:
    file_content = repo.get_contents(path)

    if isinstance(file_content, list):
        file_content = file_content[0]

    logger.info(f"Got file: {path}")
    content = (path, file_content.decoded_content.decode("utf-8"))
    return content


def _write_to_file(file_path: str, dataset: list[str]):
    with open(file_path, "w") as file:
        for data in dataset:
            file.write(f"{data}\n")


if __name__ == "__main__":
    collect_dataset()
