from pathlib import Path

import torch.nn as nn
import torch
import torch.nn.utils.rnn as rnn_utils

from src.utils import new_logger


class CodeAutocompleteRNN(nn.Module):
    """RNN-based model for code autocompletion using GRU architecture.

    This model uses a bidirectional GRU with embedding layer to predict
    the next token in a code sequence.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pad_token_id: int,
    ) -> None:
        super(CodeAutocompleteRNN, self).__init__()
        self.pad_token_id = pad_token_id
        self.logger = new_logger(__name__)

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self.pad_token_id)
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_dim * 2, vocab_size)

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.info(
            f"Model created with a total of {total_params/1000000:.0f}m trainable parameters"
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Process a batch of sequences through the model.

        Args:
            input_ids (torch.Tensor): Batch of token IDs with shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Logits for next token prediction with shape
                         (batch_size, sequence_length, vocab_size)
        """

        embedded_input = self.embedding(input_ids)

        # lenghts contain the lenght of each sequence in the batch
        lengths = (input_ids != self.pad_token_id).sum(dim=1).cpu()
        packed_input = rnn_utils.pack_padded_sequence(
            embedded_input, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.gru(packed_input)
        output, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)
        logits = self.classifier(output)
        return logits

    def predict_next_token_id(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Predict the next token ID for the given input sequence of token IDs.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape (1, sequence_length)

        Returns:
            torch.Tensor: Predicted next token ID
        """

        with torch.no_grad():
            logits = self(input_ids)

        next_token_id = (
            logits[0, -1].argmax(dim=-1).item()
        )  # I'm assuming to have a single batch and a single sequence for inference
        return next_token_id

    def save_to(self, path: Path) -> None:
        """Save the model weights and configuration to the specified path.

        Args:
            path (Path): Directory path where the model should be saved.
                        Will be created if it doesn't exist.

        Returns:
            None
        """

        directory = path.parent
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        self.logger.info(f"Model saved to {path}")

    def load_weights_from(self, path: Path) -> None:
        """Load model weights from the specified path.

        Args:
            path (Path): Path to the saved model weights

        Raises:
            FileNotFoundError: If the model file does not exist
        """

        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")

        self.load_state_dict(torch.load(path, weights_only=True))
        self.logger.info(f"Model loaded from {path}")
