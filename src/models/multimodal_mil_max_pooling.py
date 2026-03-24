class MultiModalMILMax(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, hes, ihc):
        # HES
        h_hes = self.encoder(hes)
        hes_pool, _ = torch.max(h_hes, dim=0)

        # IHC
        h_ihc = self.encoder(ihc)
        ihc_pool, _ = torch.max(h_ihc, dim=0)

        fused = torch.cat([hes_pool, ihc_pool], dim=0)

        return self.classifier(fused).squeeze()