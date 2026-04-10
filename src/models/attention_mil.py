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
    
class AttentionMIL_Papagoras(nn.Module):
    """
    from paper : https://www.biorxiv.org/content/10.64898/2025.12.19.695372v1.abstract
    Attention-based MIL model (binary head for OvR):
    - feature_extractor: Linear -> ReLU (+ optional Dropout) to get per-tile embeddings.
    - attention: Tanh-based attention producing a weight per tile.
    - classifier: Linear mapping from aggregated embedding to 2 logits (binary).
    Forward:
        Input: bag [n_tiles, input_dim]
        Output: logits [2], attention_weights [n_tiles]
    """
    def __init__(self, input_dim, hidden_dim, n_classes=2, dropout_rate=0.0):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        self.feature_extractor = nn.Sequential(*layers)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)
    def forward(self, bag):
        """
        Compute attention-pooled representation and logits.
        Args:
            bag (Tensor): [n_tiles, input_dim]
        Returns:
            logits (Tensor): [2] raw class scores.
            attn (Tensor): [n_tiles] attention weights summing to 1.
        """
        H = self.feature_extractor(bag)
        A = self.attention(H)
        A = torch.softmax(A, dim=0)
        M = torch.sum(A * H, dim=0)
        logits = self.classifier(M)
        return logits, A.squeeze(-1)