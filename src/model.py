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
        # This step is used to reduce the model size
        self.projection = nn.Linear(self.embedding.config.hidden_size, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
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

        embeddings = self.projection(embeddings)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        lstm_out, _ = self.lstm(embeddings)
        lstm_out = self.dropout(lstm_out[:, -1, :])
        logits = self.fc(lstm_out)
        return logits
