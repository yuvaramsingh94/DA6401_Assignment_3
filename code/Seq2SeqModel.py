import torch
import torch.nn as nn

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from Config import Config
from lightning import LightningModule
from lightning import Trainer, seed_everything
from RecursiveNetwork import (
    EncoderNetwork,
    DecoderNetwork,
    RNNAttentionDecoder,
    LSTMAttenDecoder,
    GRUAttenDecoder,
)


class Seq2SeqModel(LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.encoder = EncoderNetwork(self.config)
        if not self.config.attention_model:
            self.decoder = DecoderNetwork(self.config)
        else:
            if self.config.recurrent_layer_type == "RNN":
                self.decoder = RNNAttentionDecoder(self.config)
            elif self.config.recurrent_layer_type == "LSTM":
                self.decoder = LSTMAttenDecoder(self.config)
            elif self.config.recurrent_layer_type == "GRU":
                self.decoder = GRUAttenDecoder(self.config)
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.config.Y_padding_idx  # Mask out padding positions
        )

        self.train_correct = 0
        self.train_total = 0
        self.val_correct = 0
        self.val_total = 0
        self.val_exact_word_correct = 0

        self.train_loss = []
        self.val_loss = []

    def forward(self, x, X_len, y_dec_ip):
        # Encoder forward (optionally use X_len for packing)
        encoder_outputs, encoder_hidden = self.encoder(x, X_len)
        if not self.config.attention_model:

            # Decoder forward
            logits, _ = self.decoder(y_dec_ip, encoder_hidden)
            return logits
        else:
            ## encoder model
            encoder_mask = (
                x != self.config.X_padding_idx
            ).int()  # shape (batch, src seg len)
            logits = self.decoder(
                y_dec_ip, encoder_outputs, encoder_hidden, encoder_mask
            )
            return logits

    def training_step(self, batch):

        x, y_dec_ip, y_dec_op, X_len, _, _ = batch
        X_len = X_len.cpu().long()
        logits = self(x, X_len, y_dec_ip)  # (batch, tgt_len, vocab_size)
        if isinstance(logits, tuple):
            (logits, hidden, attn_weight_list) = logits
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

        x, y_dec_ip, y_dec_op, X_len, Y_len, _ = batch
        batch_size = y_dec_op.size(0)
        seq_length = y_dec_op.size(1)
        X_len = X_len.cpu().long()
        logits = self(x, X_len, y_dec_ip)  # (batch, tgt_len, vocab_size)
        if isinstance(logits, tuple):
            (logits, hidden, attn_weight_list) = logits
        ## reshaping to match the required shape of (N,C) for logits
        ## and (N,) for label
        logits = logits.view(-1, logits.size(-1))
        targets = y_dec_op.view(-1)  ## Flatten the decoder
        loss = self.loss_fn(logits, targets)

        ## Accuracy and loss tracking
        prob = F.softmax(logits, dim=1)
        preds = torch.argmax(prob, dim=1)
        correct = (preds == targets).sum().item()
        # batch_size = logits.size(0)

        ## Accuracy at the word level
        preds = preds.view(batch_size, seq_length)

        # Create mask for padding (1 for real tokens, 0 for padding)
        mask = y_dec_op != self.config.Y_padding_idx

        # Compare predictions with targets using the mask
        # A sequence is correct only if all its tokens match
        correct_sequences = 0
        for i in range(batch_size):
            # Get actual sequence length (excluding padding)
            seq_len = Y_len[i]

            # Extract predictions and targets for this sequence
            pred_seq = preds[i, :seq_len]
            target_seq = y_dec_op[i, :seq_len]

            # Check if the entire sequence matches
            if torch.all(pred_seq == target_seq):
                correct_sequences += 1

        # Update counters
        self.val_exact_word_correct += correct_sequences
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

        epoch_exact_acc = self.val_exact_word_correct / self.val_total
        self.log("val_word_acc_epoch", epoch_exact_acc)
        if len(self.val_loss) > 0:
            self.log("val_loss_epoch", torch.cat(self.val_loss).mean())
        # Reset lists
        self.val_correct = 0
        self.val_total = 0
        self.val_exact_word_correct = 0
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
