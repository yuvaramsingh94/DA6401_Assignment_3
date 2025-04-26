class Config:
    def __init__(self):
        self.wandb_project = "assignment_3"
        self.wandb_entity = "v1"
        self.epoch = 5
        self.batch_size = 16
        self.encoder_embedding_size = 128
        self.X_vocab_size = (
            26 + 1
        )  ## Here +1 is because the nn.Embedding layer throws this error AssertionError =  Padding_idx must be within num_embeddings
        self.X_padding_idx = 26
        self.X_max_length = 30
        self.Y_max_length = 26
        self.recurrent_layer_type = "RNN"
        self.encoder_hidden_size = 256
        ## For now will use same number of enc and decoder
        ## This is to avoid the hidden state mismatch between
        ## Encoder and decoder
        ## The actual solution is to use linear layer to adjust the required
        ## shape but it will add additional training parameters
        self.num_encoder_layers = 1
        self.encoder_dropout_prob = 0.0
        self.encoder_bidir = False
        self.encoder_nonlinearity = "tanh"  ## Only for RNN
        self.Y_vocab_size = (
            48 + 1
        )  ## Here +1 is because the nn.Embedding layer throws this error AssertionError =  Padding_idx must be within num_embeddings
        ## Same hidden size mismatch error between encoder and decoder
        self.decoder_embedding_size = 128
        self.Y_padding_idx = 48
        self.decoder_hidden_size = 256
        ## same value as num_encoder_layers
        self.num_decoder_layers = 1
        self.decoder_dropout_prob = 0.0
        self.decoder_bidir = False
        self.decoder_nonlinearity = "tanh"  ## Only for RNN
        self.Y_true_vocab_size = (
            48  ## No need for extra digit for padding s required by nn.Embedding
        )
        self.LR = 1e-3
