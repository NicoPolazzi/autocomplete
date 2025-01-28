import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch


class AutocompleteModel(nn.Module):
    # 768 is the size of the BERT embeddings, and 30522 is the number of tokens in the vocabulary
    def __init__(
        self,
        input_size: int = 768,
        hidden_size: int = 256,
        output_size: int = 30522,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super(AutocompleteModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.attention = _AttentionLayer(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        lstm_out, _ = self.lstm(x, (h0, c0))
        normalized = self.layer_norm(lstm_out)
        attended = self.attention(normalized)
        out = self.fc(attended)

        return out


class _AttentionLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super(_AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        attention_weights = F.softmax(self.attention(x), dim=1)
        return torch.sum(attention_weights * x, dim=1)
