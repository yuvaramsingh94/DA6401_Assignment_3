import os

## ATTENTION

"""
class Config:
    def __init__(self):
        self.wandb_project = "assignment_3"
        self.wandb_entity = "v6_attention_full_v2"
        self.epoch = 10
        self.batch_size = 64
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
        self.encoder_dropout_prob = 0.2
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
        self.LR = 0.0003246198417066587
        self.attention_size = 128
        self.attention_model = True
        self.dirpath = os.path.join("weights", "attention")
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)
        self.filename = "test"

        ##### Inference #####
        weight_base = os.path.join("weights", "Attention")
        self.WT_PATH = os.path.join(weight_base, "v2_best.ckpt")
        self.prediction_path = os.path.join(
            "predictions_attention",
        )
        self.attn_map_path = os.path.join(self.prediction_path, "attention_map")

        if not os.path.exists(self.attn_map_path):
            os.makedirs(self.attn_map_path)

        self.PRED_CSV = os.path.join(self.prediction_path, "attention_prediction.csv")

        # ## Kaggle
        # self.WT_PATH = os.path.join(
        #     "/kaggle/input/dl-assignment-3-attention-v1", "attention.ckpt"
        # )
        # self.PRED_CSV = os.path.join("weights", "attention_prediction.csv")


"""


## Basic
class Config:
    def __init__(self):
        self.wandb_project = "assignment_3"
        self.wandb_entity = "v2_basic_full"
        self.epoch = 20
        self.batch_size = 32
        self.encoder_embedding_size = 64
        self.X_vocab_size = (
            26 + 1
        )  ## Here +1 is because the nn.Embedding layer throws this error AssertionError =  Padding_idx must be within num_embeddings
        self.X_padding_idx = 26
        self.X_max_length = 30
        self.Y_max_length = 27
        self.recurrent_layer_type = "GRU"
        self.encoder_hidden_size = 512
        ## For now will use same number of enc and decoder
        ## This is to avoid the hidden state mismatch between
        ## Encoder and decoder
        ## The actual solution is to use linear layer to adjust the required
        ## shape but it will add additional training parameters
        self.num_encoder_layers = 4
        self.encoder_dropout_prob = 0.2
        self.encoder_bidir = False
        self.encoder_nonlinearity = "tanh"  ## Only for RNN
        self.Y_vocab_size = (
            48 + 1
        )  ## Here +1 is because the nn.Embedding layer throws this error AssertionError =  Padding_idx must be within num_embeddings
        ## Same hidden size mismatch error between encoder and decoder
        self.decoder_embedding_size = 64
        self.Y_padding_idx = 48
        self.decoder_hidden_size = 512
        ## same value as num_encoder_layers
        self.num_decoder_layers = 4
        self.decoder_dropout_prob = 0.5
        self.decoder_bidir = False
        self.decoder_nonlinearity = "tanh"  ## Only for RNN
        self.Y_true_vocab_size = (
            48  ## No need for extra digit for padding s required by nn.Embedding
        )
        self.LR = 0.0003000071689031731
        self.attention_size = 128
        self.attention_model = False
        self.dirpath = os.path.join("weights", "basic")
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)

        self.prediction_path = os.path.join(
            "predictions_vanilla",
        )
        if not os.path.exists(self.prediction_path):
            os.makedirs(self.prediction_path)
        self.filename = "test"

        ##### Inference #####
        weight_base = os.path.join("weights", "basic")

        self.WT_PATH = os.path.join(weight_base, "best.ckpt")
        self.PRED_CSV = os.path.join(self.prediction_path, "basic_prediction.csv")

        ## Kaggle
        # self.WT_PATH = os.path.join(
        #     "/kaggle/input/dl-assignment-3-basic-v1", "basic.ckpt"
        # )
        # self.PRED_CSV = os.path.join("weights", "basic_prediction.csv")
