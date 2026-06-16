import torch
import torch.nn as nn


class MILMeanPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        # x: [N, D]
        h = self.encoder(x)              # [N, H]
        bag_embedding = h.mean(dim=0)    # [H]
        return self.classifier(bag_embedding)  # [n_classes]
