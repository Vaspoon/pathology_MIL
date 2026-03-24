import torch
import torch.nn.functional as F

class TrainerMultiModal:
    def __init__(self, model, optimizer, device, writer=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.writer = writer

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0

        for step, (hes, ihc, y) in enumerate(loader):
            hes = hes[0].to(self.device)
            ihc = ihc[0].to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            pred = self.model(hes, ihc)

            loss = F.binary_cross_entropy_with_logits(pred, y)
            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

            # LOG STEP
            if self.writer is not None:
                global_step = epoch * len(loader) + step
                self.writer.add_scalar("train/loss_step", loss.item(), global_step)

        avg_loss = total_loss / len(loader)

        # LOG EPOCH
        if self.writer is not None:
            self.writer.add_scalar("train/loss_epoch", avg_loss, epoch)

        return avg_loss
    def eval_epoch(self, loader, epoch):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for hes, _, y in loader:
                hes = hes[0].to(self.device)
                y = y.to(self.device)

                pred = self.model(hes)
                loss = F.binary_cross_entropy_with_logits(pred, y)

                total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        if self.writer:
            self.writer.add_scalar("val/loss", avg_loss, epoch)

        return avg_loss


class Trainer:
    def __init__(self, model, optimizer, device, writer=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.writer = writer

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0

        for step, (hes, y) in enumerate(loader):
            hes = hes[0].to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            pred = self.model(hes)
            pred = pred.unsqueeze(0)

            loss = F.binary_cross_entropy_with_logits(pred, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if self.writer:
                global_step = epoch * len(loader) + step
                self.writer.add_scalar("train/loss_step", loss.item(), global_step)

        avg_loss = total_loss / len(loader)

        if self.writer:
            self.writer.add_scalar("train/loss_epoch", avg_loss, epoch)

        return avg_loss

    def eval_epoch(self, loader, epoch):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for hes, y in loader:
                hes = hes[0].to(self.device)
                y = y.to(self.device)

                pred = self.model(hes)
                pred = pred.unsqueeze(0)
                loss = F.binary_cross_entropy_with_logits(pred, y)
                total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        if self.writer:
            self.writer.add_scalar("val/loss", avg_loss, epoch)

        return avg_loss