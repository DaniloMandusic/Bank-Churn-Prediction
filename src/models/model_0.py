import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import pytorch_lightning as pl


class TabularModel(pl.LightningModule):
    def __init__(self, input_dim, lr=1e-3):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.2),

            nn.Linear(16, 1)
        )

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x).squeeze(1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()

        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()

        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = torch.sigmoid(logits) > 0.5
        acc = (preds == y.bool()).float().mean()

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()

        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = torch.sigmoid(logits) > 0.5
        acc = (preds == y.bool()).float().mean()

        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()
        return preds

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)