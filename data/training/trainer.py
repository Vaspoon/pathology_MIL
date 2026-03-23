import torch
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0

        for hes, ihc, y in loader:
            hes = hes[0].to(self.device)
            ihc = ihc[0].to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            pred = self.model(hes, ihc)

            loss = F.binary_cross_entropy_with_logits(pred, y)
            loss.backward()

            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)