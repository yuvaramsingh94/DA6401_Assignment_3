## Build the RNN network
from lightning import LightningModule
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from Config import Config

## TODO: use torch.nn.utils.rnn.pack_padded_sequence
## TODO: torch.nn.utils.rnn.pad_packed_sequence


class EncoderNetwork(LightningModule):
    def __init__(self, config: Config):
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
            e_x, lengths, batch_first=True, enforce_sorted=False
        )

        output_packed, h_n = self.recursive_layer(packed)

        output, _ = pad_packed_sequence(
            output_packed, batch_first=True, total_length=self.config.X_max_length
        )
        return output, h_n


class DecoderNetwork(nn.Module):
    def __init__(self, config: Config):
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


## Attention mechanism of RNN
class RNNAttention(nn.Module):
    def __init__(
        self, encoder_hidden_size: int, decoder_hidden_size: int, attention_size: int
    ):
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_hidden_size, attention_size, bias=False)
        self.decoder_proj = nn.Linear(decoder_hidden_size, attention_size, bias=False)
        self.v = nn.Linear(attention_size, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.tensor,  # shape (batch, 1, dec hidden)
        encoder_outputs: torch.tensor,  # shape (batch, src seq len, enc hidden)
        mask: torch.tensor,  # shape (batch, src seq len) (0 for padding index)
    ) -> tuple[torch.tensor, torch.tensor]:
        if decoder_hidden.dim() == 2:
            decoder_hidden = decoder_hidden.unsqueeze(1)
            ## Projections
            decoder_hidden_proj = self.decoder_proj(decoder_hidden)
            encoder_outputs_proj = self.encoder_proj(encoder_outputs)

            scores = self.v(
                torch.tanh(encoder_outputs_proj + decoder_hidden_proj)
            ).squeeze(-1)

            ## Mask the paddings by setting the attention weights to 0
            scores = scores.masked_fill(mask == 0, -1e9)
            attn_weights = F.softmax(scores, dim=-1)  # (batch, src seq len)
            ## batch matrix multiple [https://pytorch.org/docs/stable/generated/torch.bmm.html]
            ## If input is a ( b × n × m ) (b×n×m) tensor, mat2 is a ( b × m × p ) (b×m×p) tensor, out will be a ( b × n × p ) (b×n×p) tensor.
            context = torch.bmm(
                attn_weights.unsqueeze(1), encoder_outputs
            )  ## attn_weights = shape(batch, 1, src seq len) = output shape(batch, 1, enc_hidden)
            return context, attn_weights


"""
TODO: Compute attention weights using the previous decoder hidden state and all encoder outputs.
TODO: Get a context vector from the encoder outputs.
TODO: Concatenate the context vector with the embedded decoder input, and feed it to the RNN cell.
TODO: Use the RNN output (or hidden state) and context to produce the output logits.
"""


class RNNAttentionDecoder(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=config.Y_vocab_size,
            embedding_dim=config.decoder_embedding_size,
            padding_idx=config.Y_padding_idx,
        )
        self.rnn = nn.RNN(
            input_size=config.decoder_embedding_size + config.encoder_hidden_size,
            hidden_size=config.decoder_hidden_size,
            num_layers=config.num_decoder_layers,
            dropout=config.decoder_dropout_prob,
            bidirectional=False,  # Attention is usually used with unidirectional decoder
            nonlinearity=config.decoder_nonlinearity,
            batch_first=True,
        )
        self.attn = RNNAttention(
            encoder_hidden_size=config.encoder_hidden_size,
            decoder_hidden_size=config.decoder_hidden_size,
            attention_size=config.attention_size,
        )
        ## IF needed, send encoder attended hidden along with decoder output
        self.fc = nn.Linear(
            config.decoder_hidden_size,  # + config.encoder_hidden_size,
            config.Y_vocab_size - 1,
        )

    def forward(
        self,
        y_decoder_input: torch.tensor,  # shape (batch, tgt_seq_len)
        encoder_outputs: torch.tensor,  # shape (batch, src seq len, enc hidden)
        decoder_init_hidden: torch.tensor,  # shape (num_layers, batch, dec hidden)
        encoder_mask: torch.tensor,  # shape (batch, src seq len) (0 for padding index)
    ):
        _, tgt_seq_len = y_decoder_input.size()
        embedded = self.embedding(y_decoder_input)  # (batch, tgt_seq_len, emb_dim)

        outputs = []

        hidden = decoder_init_hidden
        attn_weights_list = []
        ## Iterate through the seguence length
        for t in range(tgt_seq_len):
            prev_hidden = hidden[-1]
            context, attn_weights = self.attn(
                prev_hidden, encoder_outputs, mask=encoder_mask
            )
            attn_weights_list.append(attn_weights.unsqueeze(1))
            emb = embedded[:, t, :]
            context = context.squeeze(1)
            rnn_input = torch.cat([emb, context], dim=1).unsqueeze(1)

            output, hidden = self.rnn(rnn_input, hidden)
            output = output.squeeze(1)  # (batch, 1, dec_hidden) -> (batch, dec_hidden)
            ## Concatenate output and context
            # out = self.fc(torch.cat([output, context], dim=1))
            out = self.fc(output)  ## only use the output from decoder
            outputs.append(out.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        attn_weights = torch.cat(
            attn_weights_list, dim=1
        )  # (batch, tgt_seq_len, src_seq_len)
        return outputs, attn_weights


class LSTMAttention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim, attn_dim):
        super().__init__()
        self.enc_proj = nn.Linear(enc_hidden_dim, attn_dim, bias=False)
        self.dec_proj = nn.Linear(dec_hidden_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.tensor,
        encoder_outputs: torch.tensor,
        mask: torch.tensor,
    ):
        # decoder_hidden: (batch, dec_hidden)
        # encoder_outputs: (batch, src_len, enc_hidden)
        src_len = encoder_outputs.size(1)
        dec_hidden = (
            self.dec_proj(decoder_hidden).unsqueeze(1).repeat(1, src_len, 1)
        )  # (batch, src_len, attn_dim)
        enc_hidden = self.enc_proj(encoder_outputs)  # (batch, src_len, attn_dim)
        energy = self.v(torch.tanh(enc_hidden + dec_hidden)).squeeze(
            -1
        )  # (batch, src_len)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(energy, dim=1)  # (batch, src_len)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(
            1
        )  # (batch, enc_hidden)
        return context, attn_weights


class LSTMAttenDecoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.embedding = nn.Embedding(
            config.Y_vocab_size,
            config.decoder_embedding_size,
            padding_idx=config.Y_padding_idx,
        )
        self.attn = LSTMAttention(
            enc_hidden_dim=config.encoder_hidden_size,
            dec_hidden_dim=config.decoder_hidden_size,
            attn_dim=config.attention_size,
        )
        self.lstm = nn.LSTM(
            input_size=config.decoder_embedding_size + config.encoder_hidden_size,
            hidden_size=config.decoder_hidden_size,
            num_layers=config.num_decoder_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(
            config.decoder_hidden_size,  # + config.encoder_hidden_size,
            config.Y_vocab_size - 1,
        )

    def forward(
        self, y_decoder_input, encoder_outputs, decoder_init, encoder_mask=None
    ):
        """
        y_decoder_input: shape (batch, tgt_seq_len)
        encoder_outputs: shape (batch, src_seq_len, enc_hidden)
        decoder_init: shape tuple (h_0, c_0) for LSTM
        encoder_mask: shape (batch, src_seq_len)
        """
        batch_size, tgt_seq_len = y_decoder_input.size()
        embedded = self.embedding(y_decoder_input)  # (batch, tgt_seq_len, emb_dim)
        h, c = decoder_init  # (num_layers, batch, dec_hidden)
        outputs = []
        attn_weights_list = []
        for t in range(tgt_seq_len):
            prev_hidden = h[-1]  # (batch, dec_hidden)
            context, attn_weights = self.attn(
                prev_hidden, encoder_outputs, mask=encoder_mask
            )
            attn_weights_list.append(attn_weights.unsqueeze(1))
            lstm_input = torch.cat([embedded[:, t, :], context], dim=1).unsqueeze(
                1
            )  # (batch, 1, emb+enc_hidden)
            output, (h, c) = self.lstm(lstm_input, (h, c))
            output = output.squeeze(1)
            # out = self.fc(torch.cat([output, context], dim=1))  # (batch, vocab_size-1)
            out = self.fc(output)  # (batch, vocab_size-1)
            outputs.append(out.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)  # (batch, tgt_seq_len, vocab_size-1)
        attn_weights = torch.cat(
            attn_weights_list, dim=1
        )  # (batch, tgt_seq_len, src_seq_len)
        return outputs, attn_weights


class GRUAttenDecoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.embedding = nn.Embedding(
            config.Y_vocab_size,
            config.decoder_embedding_size,
            padding_idx=config.Y_padding_idx,
        )
        self.attn = LSTMAttention(
            enc_hidden_dim=config.encoder_hidden_size,
            dec_hidden_dim=config.decoder_hidden_size,
            attn_dim=config.attention_size,
        )
        self.gru = nn.GRU(
            input_size=config.decoder_embedding_size + config.encoder_hidden_size,
            hidden_size=config.decoder_hidden_size,
            num_layers=config.num_decoder_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(
            config.decoder_hidden_size,  # + config.encoder_hidden_size,
            config.Y_vocab_size - 1,
        )

    def forward(
        self, y_decoder_input, encoder_outputs, decoder_init, encoder_mask=None
    ):
        """
        y_decoder_input: shape (batch, tgt_seq_len)
        encoder_outputs: shape (batch, src_seq_len, enc_hidden)
        decoder_init: shape (num_layers, batch, dec_hidden)
        encoder_mask: shape (batch, src_seq_len)
        """
        batch_size, tgt_seq_len = y_decoder_input.size()
        embedded = self.embedding(y_decoder_input)  # (batch, tgt_seq_len, emb_dim)
        h = decoder_init  # (num_layers, batch, dec_hidden)
        outputs = []
        attn_weights_list = []
        for t in range(tgt_seq_len):
            prev_hidden = h[-1]  # (batch, dec_hidden)
            context, attn_weights = self.attn(
                prev_hidden, encoder_outputs, mask=encoder_mask
            )
            attn_weights_list.append(attn_weights.unsqueeze(1))
            gru_input = torch.cat([embedded[:, t, :], context], dim=1).unsqueeze(
                1
            )  # (batch, 1, emb+enc_hidden)
            output, h = self.gru(gru_input, h)
            output = output.squeeze(1)
            # out = self.fc(torch.cat([output, context], dim=1))  # (batch, vocab_size-1)
            out = self.fc(output)  # (batch, vocab_size-1)
            outputs.append(out.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)  # (batch, tgt_seq_len, vocab_size-1)
        attn_weights = torch.cat(
            attn_weights_list, dim=1
        )  # (batch, tgt_seq_len, src_seq_len)
        return outputs, attn_weights
