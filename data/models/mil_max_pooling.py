import torch
import torch.nn as nn

class MILMaxPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [N, D]

        h = self.encoder(x)  # [N, H]

        # MAX POOLING (MIL)
        bag_embedding, _ = torch.max(h, dim=0)  # [H]

        out = self.classifier(bag_embedding)

        return out.squeeze()