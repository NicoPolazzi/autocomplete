""" Probably remove this module in the final version of the program"""

import re

import torch
from transformers import RobertaTokenizer, RobertaModel
from datasets import load_dataset

from src.logger import get_logger

MAX_SAMPLES = 100
PRETRAINED_MODEL_PATH = "microsoft/codebert-base"

logger = get_logger(__name__)


def get_embeddings_and_next_token_pairs() -> tuple[list[torch.Tensor], list[int]]:
    tokenizer = RobertaTokenizer.from_pretrained(PRETRAINED_MODEL_PATH)
    model = RobertaModel.from_pretrained(PRETRAINED_MODEL_PATH)
    model.eval()

    embeddings = []
    next_tokens = []
    batch_size = 32

    # Process in batches
    dataset = load_dataset("flytech/python-codes-25k", split=f"train[:{MAX_SAMPLES}]")
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]

        # Process batch
        code_snippets = [
            _remove_triple_backticks_and_comments(snippet) for snippet in batch["output"]
        ]
        tokenized_snippets = _tokenize_code_snippets(code_snippets, tokenizer)

        # Generate pairs and embeddings for batch
        with torch.no_grad():
            for snippet in tokenized_snippets:
                pairs = _create_input_output_pairs(snippet)
                if pairs:
                    contexts, tokens = zip(*pairs)
                    batch_embeddings = _generate_embeddings_batch(contexts, model)
                    embeddings.extend(batch_embeddings)
                    next_tokens.extend(tokens)

        logger.info(f"Processed batch {i//batch_size + 1}")

    return embeddings, next_tokens


def _generate_embeddings_batch(contexts: list[int], model: RobertaModel) -> list[torch.Tensor]:
    device = next(model.parameters()).device
    batch_tensor = torch.tensor(contexts).to(device)
    embeddings = model(batch_tensor)[0]
    return [emb for emb in embeddings.cpu()]


def _load_raw_dataset(dataset_path: str = "flytech/python-codes-25k") -> list[str]:
    dataset = load_dataset(dataset_path, split="train")
    return dataset["output"]


def _remove_triple_backticks_and_comments(code: str) -> str:
    code = re.sub(r"^```python\n", "", code)
    code = re.sub(r"```$", "", code)
    code = re.sub(r"#.*$", "", code, flags=re.MULTILINE)
    lines = [line.rstrip() for line in code.splitlines() if line.strip()]
    return "\n".join(lines)


def _tokenize_code_snippets(
    code_snippets: list[str], tokenizer: RobertaTokenizer
) -> list[list[int]]:
    return [tokenizer.encode(snippet, add_special_tokens=False) for snippet in code_snippets]


def _create_input_output_pairs(
    tokenized_snippet: list[int], context_length: int = 50
) -> list[tuple[list[int], int]]:
    pairs = []

    for i in range(len(tokenized_snippet) - context_length):
        context = tokenized_snippet[i : i + context_length]
        next_token = tokenized_snippet[i + context_length]
        pairs.append((context, next_token))

    return pairs


def _generate_embedding(context: list[int], model: RobertaModel) -> torch.Tensor:
    context_embeddings = model(torch.tensor(context)[None, :])[0]
    return context_embeddings[0]
