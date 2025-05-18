import os

## ATTENTION
"""
class Config:
    def __init__(self):
        self.wandb_project = "assignment_3"
        self.wandb_entity = "v1_attention_full"
        self.epoch = 50
        self.batch_size = 64
        self.encoder_embedding_size = 512
        self.X_vocab_size = (
            26 + 1
        )  ## Here +1 is because the nn.Embedding layer throws this error AssertionError =  Padding_idx must be within num_embeddings
        self.X_padding_idx = 26
        self.X_max_length = 30
        self.Y_max_length = 27
        self.recurrent_layer_type = "LSTM"
        self.encoder_hidden_size = 512
        ## For now will use same number of enc and decoder
        ## This is to avoid the hidden state mismatch between
        ## Encoder and decoder
        ## The actual solution is to use linear layer to adjust the required
        ## shape but it will add additional training parameters
        self.num_encoder_layers = 2
        self.encoder_dropout_prob = 0.2
        self.encoder_bidir = False
        self.encoder_nonlinearity = "tanh"  ## Only for RNN
        self.Y_vocab_size = (
            48 + 1
        )  ## Here +1 is because the nn.Embedding layer throws this error AssertionError =  Padding_idx must be within num_embeddings
        ## Same hidden size mismatch error between encoder and decoder
        self.decoder_embedding_size = 512
        self.Y_padding_idx = 48
        self.decoder_hidden_size = 512
        ## same value as num_encoder_layers
        self.num_decoder_layers = 2
        self.decoder_dropout_prob = 0.0
        self.decoder_bidir = False
        self.decoder_nonlinearity = "tanh"  ## Only for RNN
        self.Y_true_vocab_size = (
            48  ## No need for extra digit for padding s required by nn.Embedding
        )
        self.LR = 0.0008193613448607609
        self.attention_size = 128
        self.attention_model = True
        self.dirpath = os.path.join("weights", "basic")
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)
        self.filename = "test"

        ##### Inference #####
        weight_base = os.path.join("weights", "Attention")
        self.WT_PATH = os.path.join(weight_base, "attention.ckpt")
        self.PRED_CSV = os.path.join(weight_base, "prediction.csv")

        ## Kaggle
        self.WT_PATH = os.path.join(
            "/kaggle/input/dl-assignment-3-attention-v1", "attention.ckpt"
        )
        self.PRED_CSV = os.path.join("weights", "prediction.csv")
"""


## Basic
class Config:
    def __init__(self):
        self.wandb_project = "assignment_3"
        self.wandb_entity = "v1_basic_full"
        self.epoch = 50
        self.batch_size = 16
        self.encoder_embedding_size = 256
        self.X_vocab_size = (
            26 + 1
        )  ## Here +1 is because the nn.Embedding layer throws this error AssertionError =  Padding_idx must be within num_embeddings
        self.X_padding_idx = 26
        self.X_max_length = 30
        self.Y_max_length = 27
        self.recurrent_layer_type = "LSTM"
        self.encoder_hidden_size = 512
        ## For now will use same number of enc and decoder
        ## This is to avoid the hidden state mismatch between
        ## Encoder and decoder
        ## The actual solution is to use linear layer to adjust the required
        ## shape but it will add additional training parameters
        self.num_encoder_layers = 2
        self.encoder_dropout_prob = 0.4
        self.encoder_bidir = False
        self.encoder_nonlinearity = "tanh"  ## Only for RNN
        self.Y_vocab_size = (
            48 + 1
        )  ## Here +1 is because the nn.Embedding layer throws this error AssertionError =  Padding_idx must be within num_embeddings
        ## Same hidden size mismatch error between encoder and decoder
        self.decoder_embedding_size = 256
        self.Y_padding_idx = 48
        self.decoder_hidden_size = 512
        ## same value as num_encoder_layers
        self.num_decoder_layers = 2
        self.decoder_dropout_prob = 0.0
        self.decoder_bidir = False
        self.decoder_nonlinearity = "tanh"  ## Only for RNN
        self.Y_true_vocab_size = (
            48  ## No need for extra digit for padding s required by nn.Embedding
        )
        self.LR = 0.0008193613448607609
        self.attention_size = 128
        self.attention_model = False
        self.dirpath = os.path.join("weights", "basic")
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)
        self.filename = "test"

        ##### Inference #####
        weight_base = os.path.join("weights", "basic")
        self.WT_PATH = os.path.join(weight_base, "basic.ckpt")
        self.PRED_CSV = os.path.join(weight_base, "prediction.csv")

        ## Kaggle
        # self.WT_PATH = os.path.join(
        #     "/kaggle/input/dl-assignment-3-basic-v1", "basic.ckpt"
        # )
        # self.PRED_CSV = os.path.join("weights", "prediction.csv")
