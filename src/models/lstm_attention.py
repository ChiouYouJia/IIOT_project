"""LSTM classifier with learned temporal attention."""

import torch
from torch import nn


class LSTMAttentionClassifier(nn.Module):
    """Encode a causal sequence and attend over its hidden states."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        dropout: float = 0.2,
        lstm_dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        hidden_states, _ = self.lstm(x)
        attention_logits = self.attention(hidden_states).squeeze(-1)
        attention_weights = torch.softmax(attention_logits, dim=1)
        context = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=1)
        return self.classifier(self.dropout(context))


def build_model(input_size: int, num_classes: int, hidden_size: int = 128,
                dropout: float = 0.2, lstm_dropout: float = 0.0):
    return LSTMAttentionClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_classes=num_classes,
        dropout=dropout,
        lstm_dropout=lstm_dropout,
    )
