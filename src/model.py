import torch.nn as nn
import torch


class CodeRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        model_type: str = "LSTM",
    ) -> None:
        super(CodeRNN, self).__init__()
        self.hidden_size = hidden_size
        self.model_type = model_type
        self.rnn: nn.Module

        if model_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif model_type == "GRU":
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        else:
            raise ValueError("model_type should be either 'LSTM' or 'GRU'")

        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        if self.model_type == "LSTM":
            output, (hidden, _) = self.rnn(x)
        elif self.model_type == "GRU":
            output, hidden = self.rnn(x)

        output = self.fc(output[:, -1, :])
        output = self.softmax(output)
        return output
