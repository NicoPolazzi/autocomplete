import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch
from src.logger import get_logger
from transformers import AutoModel

logger = get_logger(__name__)


class CodeAutocompleteModel(nn.Module):
    def __init__(self, codebert_model_name="microsoft/codebert-base", hidden_dim=256, num_layers=2, vocab_size=None):
        super(CodeAutocompleteModel, self).__init__()
        self.codebert = AutoModel.from_pretrained(codebert_model_name)
        self.lstm = nn.LSTM(
            input_size=self.codebert.config.hidden_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            embeddings = self.codebert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        lstm_out, _ = self.lstm(embeddings)
        logits = self.fc(lstm_out[:, -1, :])
        return logits
