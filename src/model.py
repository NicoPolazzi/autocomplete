import torch.nn as nn
import torch
from transformers import AutoModel

from src.logger import get_logger

logger = get_logger(__name__)

MODEL_NAME = "microsoft/codebert-base"


class CodeAutocompleteModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout_rate: float = 0.3,
    ):
        super(CodeAutocompleteModel, self).__init__()
        self.embedding = AutoModel.from_pretrained(MODEL_NAME)
        self.batch_norm = nn.BatchNorm1d(self.embedding.config.hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(
            input_size=self.embedding.config.hidden_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        with torch.no_grad():
            embeddings = self.embedding(input_ids, attention_mask).last_hidden_state

        embeddings = self.batch_norm(embeddings.transpose(1, 2)).transpose(1, 2)
        embeddings = self.dropout(embeddings)
        lstm_out, _ = self.lstm(embeddings)
        lstm_out = self.dropout(lstm_out[:, -1, :])
        logits = self.fc(lstm_out)
        return logits
