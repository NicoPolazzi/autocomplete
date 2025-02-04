import torch.nn as nn
import torch
from transformers import AutoModel

from src.logger import get_logger
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

logger = get_logger(__name__)

MODEL_NAME = "microsoft/codebert-base"


class CodeAutocompleteModel(nn.Module):
    def __init__(self, vocab_size: int):
        super(CodeAutocompleteModel, self).__init__()
        self.embedding = AutoModel.from_pretrained(MODEL_NAME)
        for param in self.embedding.parameters():
            param.requires_grad = False

        hidden_size = self.embedding.config.hidden_size
        self.classifier = nn.Linear(hidden_size, vocab_size)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.embedding(input_ids, attention_mask)
        hidden_states = outputs.last_hidden_state
        logits = self.classifier(hidden_states)
        return logits


class CodeAutocompleteRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.1):
        super(CodeAutocompleteRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout
        )
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        x = self.embedding(input_ids)
        lengths = attention_mask.sum(dim=1).to("cpu")

        # Pack the sequences so the LSTM ignores pad tokens.
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_x)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=x.size(1))

        logits = self.classifier(output)
        return logits
