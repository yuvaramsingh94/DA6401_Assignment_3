## Build the RNN network
from lightning import LightningModule
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


## TODO: use torch.nn.utils.rnn.pack_padded_sequence
## TODO: torch.nn.utils.rnn.pad_packed_sequence


class EncoderNetwork(LightningModule):
    def __init__(self, config: dict):
        super(EncoderNetwork, self).__init__()
        self.config = config
        ## encoder
        ### Embedding layer
        #### Here config.X_vocab_size will be the padding index also
        self.embedding = nn.Embedding(
            self.config.X_vocab_size,
            self.config.encoder_embedding_size,
            padding_idx=self.config.X_padding_idx,
        )

        if self.config.recurrent_layer_type == "RNN":
            self.recursive_layer = nn.RNN(
                input_size=self.config.encoder_embedding_size,
                hidden_size=self.config.encoder_hidden_size,
                num_layers=self.config.num_encoder_layers,
                dropout=self.config.encoder_dropout_prob,
                bidirectional=self.config.encoder_bidir,
                nonlinearity=self.config.encoder_nonlinearity,
                batch_first=True,
            )
        elif self.config.recurrent_layer_type == "LSTM":
            self.recursive_layer = nn.LSTM(
                input_size=self.config.encoder_embedding_size,
                hidden_size=self.config.encoder_hidden_size,
                num_layers=self.config.num_encoder_layers,
                dropout=self.config.encoder_dropout_prob,
                bidirectional=self.config.encoder_bidir,
                batch_first=True,
            )
        elif self.config.recurrent_layer_type == "GRU":
            self.recursive_layer = nn.GRU(
                input_size=self.config.encoder_embedding_size,
                hidden_size=self.config.encoder_hidden_size,
                num_layers=self.config.num_encoder_layers,
                dropout=self.config.encoder_dropout_prob,
                bidirectional=self.config.encoder_bidir,
                batch_first=True,
            )

    def forward(
        self, x: torch.tensor, lengths: torch.tensor
    ) -> tuple[torch.tensor, torch.tensor]:
        ## Initialize H0
        ##! The doc said the H0 will dafault to zeros. Going to check this https://pytorch.org/docs/stable/generated/torch.nn.RNN.html#torch.nn.RNN
        e_x = self.embedding(x)

        ## Pack the padded input for better computation
        packed = pack_padded_sequence(
            e_x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        output_packed, h_n = self.recursive_layer(packed)

        output, _ = pad_packed_sequence(
            output_packed, batch_first=True, total_length=self.config.X_max_length
        )
        return output, h_n


class DecoderNetwork(nn.Module):
    def __init__(self, config: dict):
        super(DecoderNetwork, self).__init__()
        self.config = config
        ## Tamil encoder
        self.embedding = nn.Embedding(
            num_embeddings=config.Y_vocab_size,
            embedding_dim=config.decoder_embedding_size,
            padding_idx=config.Y_padding_idx,
        )

        if self.config.recurrent_layer_type == "RNN":
            self.recursive_layer = nn.RNN(
                input_size=self.config.decoder_embedding_size,
                hidden_size=self.config.decoder_hidden_size,
                num_layers=self.config.num_decoder_layers,
                dropout=self.config.decoder_dropout_prob,
                bidirectional=self.config.decoder_bidir,
                nonlinearity=self.config.decoder_nonlinearity,
                batch_first=True,
            )
        elif self.config.recurrent_layer_type == "LSTM":
            self.recursive_layer = nn.LSTM(
                input_size=self.config.decoder_embedding_size,
                hidden_size=self.config.decoder_hidden_size,
                num_layers=self.config.num_decoder_layers,
                dropout=self.config.decoder_dropout_prob,
                bidirectional=self.config.decoder_bidir,
                batch_first=True,
            )
        elif self.config.recurrent_layer_type == "GRU":
            self.recursive_layer = nn.GRU(
                input_size=self.config.decoder_embedding_size,
                hidden_size=self.config.decoder_hidden_size,
                num_layers=self.config.num_decoder_layers,
                dropout=self.config.decoder_dropout_prob,
                bidirectional=self.config.decoder_bidir,
                batch_first=True,
            )
        ## Here the Vocab size should be one less as we added 1 for the embedding layer
        self.fc = nn.Linear(config.decoder_hidden_size, config.Y_vocab_size - 1)

    def forward(
        self, y_decoder_input: torch.tensor, encoder_hidden: torch.tensor
    ) -> tuple[torch.tensor, torch.tensor]:
        """
        y_decoder_input: (batch, tgt_seq_len)
        encoder_hidden: (num_layers * num_directions, batch, hidden_size)
        """
        embedded = self.embedding(y_decoder_input)  # (batch, tgt_seq_len, embed_dim)
        output, hidden = self.recursive_layer(embedded, encoder_hidden)
        # output: (batch, tgt_seq_len, hidden_size)
        logits = self.fc(output)  # (batch, tgt_seq_len, vocab_size)
        return logits, hidden
