import torch.nn as nn
from .attention_mil import AttentionMIL
import torch

class MultiModalMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.hes_mil = AttentionMIL(input_dim, hidden_dim)
        self.ihc_mil = AttentionMIL(input_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, hes, ihc):
        h1, _ = self.hes_mil(hes)
        h2, _ = self.ihc_mil(ihc)

        fused = torch.cat([h1, h2], dim=0)

        return self.classifier(fused).squeeze()