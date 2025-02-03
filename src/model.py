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
    ):
        super(CodeAutocompleteModel, self).__init__()
        self.embedding = AutoModel.from_pretrained(MODEL_NAME)
        self.lstm = nn.LSTM(
            input_size=self.embedding.config.hidden_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        with torch.no_grad():
            embeddings = self.embedding(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
        lstm_out, _ = self.lstm(embeddings)
        logits = self.fc(lstm_out[:, -1, :])
        return logits
