import torch
import torch.nn as nn

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger

from lightning import LightningModule
from lightning import Trainer, seed_everything
from RecursiveNetwork import EncoderNetwork, DecoderNetwork


class Seq2SeqModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = EncoderNetwork(self.config)
        self.decoder = DecoderNetwork(self.config)
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.config.Y_padding_idx  # Mask out padding positions
        )

        self.train_correct = 0
        self.train_total = 0
        self.val_correct = 0
        self.val_total = 0

        self.train_loss = []
        self.val_loss = []

    def forward(self, x, X_len, y_dec_ip):
        # Encoder forward (optionally use X_len for packing)
        _, encoder_hidden = self.encoder(x, X_len)
        # Decoder forward
        logits, _ = self.decoder(y_dec_ip, encoder_hidden)
        return logits

    def training_step(self, batch):

        x, y_dec_ip, y_dec_op, X_len, _, _ = batch

        logits = self(x, X_len, y_dec_ip)  # (batch, tgt_len, vocab_size)
        ## reshaping to match the required shape of (N,C) for logits
        ## and (N,) for label
        logits = logits.view(-1, logits.size(-1))
        targets = y_dec_op.view(-1)  ## Flatten the decoder
        loss = self.loss_fn(logits, targets)

        ## Accuracy and loss tracking
        prob = F.softmax(logits, dim=1)
        preds = torch.argmax(prob, dim=1)
        correct = (preds == targets).sum().item()
        batch_size = logits.size(0)

        # Update counters
        self.train_correct += correct
        self.train_total += batch_size
        self.train_loss.append(loss.view(1).cpu())

        # self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):

        x, y_dec_ip, y_dec_op, X_len, _, _ = batch

        logits = self(x, X_len, y_dec_ip)  # (batch, tgt_len, vocab_size)
        ## reshaping to match the required shape of (N,C) for logits
        ## and (N,) for label
        logits = logits.view(-1, logits.size(-1))
        targets = y_dec_op.view(-1)  ## Flatten the decoder
        loss = self.loss_fn(logits, targets)

        ## Accuracy and loss tracking
        prob = F.softmax(logits, dim=1)
        preds = torch.argmax(prob, dim=1)
        correct = (preds == targets).sum().item()
        batch_size = logits.size(0)

        # Update counters
        self.val_correct += correct
        self.val_total += batch_size
        self.val_loss.append(loss.view(1).cpu())

        # self.log("val_loss", loss)
        return loss

    def on_train_epoch_end(self):
        # Calculate epoch accuracy
        epoch_acc = self.train_correct / self.train_total
        self.log("train_acc_epoch", epoch_acc)
        if len(self.train_loss) > 0:
            self.log("train_loss_epoch", torch.cat(self.train_loss).mean())
        # Reset lists
        self.train_correct = 0
        self.train_total = 0
        self.train_loss = []

    def on_validation_epoch_end(self):
        # Calculate epoch accuracy
        epoch_acc = self.val_correct / self.val_total
        self.log("val_acc_epoch", epoch_acc)
        if len(self.val_loss) > 0:
            self.log("val_loss_epoch", torch.cat(self.val_loss).mean())
        # Reset lists
        self.val_correct = 0
        self.val_total = 0
        self.val_loss = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.LR)
        lr_scheduler_config = {
            "scheduler": ReduceLROnPlateau(
                optimizer=optimizer, mode="max", factor=0.1, patience=2
            ),
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_acc_epoch",
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": True,
            "name": "LR_track",
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
