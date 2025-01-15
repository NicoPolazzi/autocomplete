import re

from logger import get_logger
from dataset import create_dataloader

logger = get_logger(__name__)

DATASET_PATH = "data/python_files.txt"


def preprocess_data():
    code = _read_code_from(DATASET_PATH)
    cleaned_code = _clean_code(code)
    tokens = _tokenize_code(cleaned_code)
    return tokens


def _read_code_from(file_path: str) -> str:
    with open(file_path, "r") as file:
        code = file.read()
    return code


def _clean_code(code: str) -> str:
    # Remove docstrings (both single and multi-line)
    code = re.sub(r'""".*?"""', "", code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", "", code, flags=re.DOTALL)
    # Normalize tabs to 4 spaces
    code = re.sub(r"\t", "    ", code)
    # Remove single-line comments
    code = re.sub(r"#.*", "", code)
    # Remove blank lines
    code = re.sub(r"\n\s*\n", "\n", code)
    # Remove imports
    code = re.sub(r"^\s*(import|from)\s+.*$", "", code, flags=re.MULTILINE)
    # Remove leading and trailing whitespace
    code = code.strip()
    # Remove multiple whitespace characters
    code = re.sub(r"\s+", " ", code)
    return code


def _tokenize_code(code: str) -> list[str]:
    tokens = re.findall(r'\w+|"[^"]*"|\'[^\']*\'|[^\w\s]', code)
    return tokens


def create_vocabulary(tokens: list[str]) -> dict:
    vocab = {token: idx for idx, token in enumerate(set(tokens))}
    return vocab


if __name__ == "__main__":
    tokens = preprocess_data()
    vocab = create_vocabulary(tokens)
    dataloader = create_dataloader(tokens, vocab, batch_size=32)

    print(f"First few tokens: {tokens[:10]}")

    # Print the vocabulary
    print(f"Vocabulary: {vocab.keys()}")

    for i, batch in enumerate(dataloader):
        print(f"Batch {i+1} size: {batch.size()}")
        print(f"Batch {i+1} content: {batch}")
        if i >= 2:  # Print only the first 3 batches for demonstration
            break
