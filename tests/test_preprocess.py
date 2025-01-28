import torch
from transformers import RobertaTokenizer, RobertaModel
from src.preprocess import (
    _generate_embeddings_batch,
    _remove_triple_backticks_and_comments,
    _tokenize_code_snippets,
    _create_input_output_pairs,
    _generate_embedding,
)


def test_remove_triple_backticks_and_comments():
    code = """```python
    # comment
    def hello(): 
        print('world')
    ```"""
    expected = """    def hello():
        print('world')"""

    result = _remove_triple_backticks_and_comments(code)
    assert result == expected


def test_tokenize_code_snippets():
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    code_snippets = ["def test():", "    print('hello')"]

    tokens = _tokenize_code_snippets(code_snippets, tokenizer)

    assert len(tokens) == 2
    assert all(isinstance(t, list) for t in tokens)
    assert all(isinstance(token, int) for t in tokens for token in t)


def test_create_input_output_pairs():
    tokenized_snippet = list(range(100))
    context_length = 50

    pairs = _create_input_output_pairs(tokenized_snippet, context_length)

    assert len(pairs) == len(tokenized_snippet) - context_length

    for context, next_token in pairs:
        assert len(context) == context_length
        assert isinstance(next_token, int)


def test_generate_embedding():
    model = RobertaModel.from_pretrained("microsoft/codebert-base")
    token_ids = [1, 2, 3, 4, 5]

    embedding = _generate_embedding(token_ids, model)

    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape[0] == len(token_ids)
    assert embedding.shape[1] == 768


def test_generate_embeddings_batch():
    model = RobertaModel.from_pretrained("microsoft/codebert-base")
    # Create multiple contexts
    contexts = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    batch_embeddings = _generate_embeddings_batch(contexts, model)

    # Test output format
    assert isinstance(batch_embeddings, list)
    assert len(batch_embeddings) == len(contexts)

    # Test each embedding
    for emb in batch_embeddings:
        assert isinstance(emb, torch.Tensor)
        assert emb.shape[0] == 3  # sequence length
        assert emb.shape[1] == 768  # CodeBERT hidden size
