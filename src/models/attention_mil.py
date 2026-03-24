import torch
import torch.nn as nn

class AttentionMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        h = self.encoder(x)
        A = self.attention(h)
        A = torch.softmax(A, dim=0)
        bag = torch.sum(A * h, dim=0)

        return bag, A