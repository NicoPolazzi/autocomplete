import torch.nn as nn
import torch

from src.logger import new_logger


class CodeAutocompleteRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ) -> None:
        super(CodeAutocompleteRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout
        )
        self.classifier = nn.Linear(hidden_dim, vocab_size)

        logger = new_logger(__name__)
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Model created with a total of {total_params} trainable parameters")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        lengths = attention_mask.sum(dim=1).long()
        embedded_input = self.embedding(input_ids)

        lstm_output, _ = self.lstm(embedded_input)
        batch_size = lstm_output.size(0)

        last_hidden_state = lstm_output[torch.arange(batch_size), lengths - 1, :]
        logits = self.classifier(last_hidden_state)

        return logits
