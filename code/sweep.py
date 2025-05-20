from Config import Config
from torch.utils.data import DataLoader
import torch
import wandb
import gc
from lightning.pytorch.loggers import WandbLogger
import os
from lightning import Trainer, seed_everything
from dataloader import CustomTextDataset
import argparse
import pandas as pd
from Seq2SeqModel import Seq2SeqModel

SEED = 5
seed_everything(SEED, workers=True)

## Get the Required keys from secrets if its in kaggle
parser = argparse.ArgumentParser(description="HP sweep")
parser.add_argument(
    "--kaggle", action="store_true", help="Set this flag to true if its kaggle"
)
parser.add_argument(
    "--colab", action="store_true", help="Set this flag to true if its colab"
)
parser.add_argument("-w", "--wandb_key", type=str, help="wandb key")
# Parse the arguments
args = parser.parse_args()


if args.kaggle:
    ## Kaggle secret
    from kaggle_secrets import UserSecretsClient

    secret_label = "wandb_api_key"
    wandb_key = UserSecretsClient().get_secret(secret_label)
    wandb.login(key=wandb_key)
elif args.colab:
    ## Kaggle secret
    # from google.colab import userdata

    # secret_label = "wandb_api_key"
    # wandb_key = userdata.get(secret_label)
    wandb.login(key=args.wandb_key)
else:
    wandb.login()


## Dataloader
DATASET_PATH = os.path.join("dataset", "dakshina_dataset_v1.0", "ta", "lexicons")
if args.kaggle:
    DATASET_PATH = os.path.join("/kaggle", "input", "intro-to-da-a3-d-v2", "lexicons")
    if not os.path.exists(DATASET_PATH):
        DATASET_PATH = os.path.join("/kaggle", "input", "lexicons")

if args.colab:
    DATASET_PATH = os.path.join(
        "/content", "DA6401_Assignment_3", "dataset", "lexicons"
    )


sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_word_acc_epoch"},
    "parameters": {
        "learning_rate": {"max": 0.001, "min": 0.0000001},
        "embedding_size": {"values": [128, 256, 512, 64]},
        "recurrent_layer_type": {"values": ["LSTM", "GRU"]},
        "hidden_size": {"values": [128, 256, 512, 64]},
        "batch_size": {"values": [16, 32, 64]},
        "num_layers": {"values": [1, 2, 4, 6]},
        # "num_layers": {"values": [1]},
        "encoder_dropout_prob": {"values": [0.0, 0.2, 0.4, 0.5]},
        "encoder_nonlinearity": {"values": ["tanh"]},
        # "decoder_embedding_size": {"values": [128, 256, 512, 64]},
        # "decoder_hidden_size": {"values": [128, 256, 512, 64]},
        # "num_decoder_layers": {"values": [2, 4, 6]},
        "decoder_dropout_prob": {"values": [0.0, 0.2, 0.4, 0.5]},
        "decoder_nonlinearity": {"values": ["tanh"]},
        "Attention": {"values": [False]},
        # "Attention_size": {"values": [128, 256, 512, 64]},
    },
    "early_terminate": {"type": "hyperband", "min_iter": 2, "eta": 2},
}

train_df = pd.read_csv(os.path.join(DATASET_PATH, "ta.translit.sampled.train.idx.csv"))
val_df = pd.read_csv(os.path.join(DATASET_PATH, "ta.translit.sampled.dev.idx.csv"))

print("Train", train_df.shape, "Val", val_df.shape)


def main():
    """
    The main function that has all the code to run a training.
    This will be used by the sweep agent to run multiple hyperparamter
    tuning.
    """
    config = Config()

    wandb.init(
        # Set the project where this run will be logged
        project=config.wandb_project,
        # Track hyperparameters and run metadata
        # config=config,
    )

    wandb.run.name = f"Basic_rec_{wandb.config.recurrent_layer_type}"

    ## Update the config instance with the hpt from sweep
    config.LR = wandb.config.learning_rate
    config.batch_size = wandb.config.batch_size
    config.encoder_embedding_size = wandb.config.embedding_size
    config.recurrent_layer_type = wandb.config.recurrent_layer_type
    config.encoder_hidden_size = wandb.config.hidden_size
    config.num_encoder_layers = wandb.config.num_layers
    config.encoder_dropout_prob = wandb.config.encoder_dropout_prob
    config.encoder_nonlinearity = wandb.config.encoder_nonlinearity
    config.decoder_embedding_size = wandb.config.embedding_size
    config.decoder_hidden_size = wandb.config.hidden_size
    config.num_decoder_layers = wandb.config.num_layers
    config.decoder_dropout_prob = wandb.config.decoder_dropout_prob
    config.decoder_nonlinearity = wandb.config.decoder_nonlinearity
    config.attention_model = wandb.config.Attention
    # config.attention_size = wandb.config.Attention_size

    train_dataset = CustomTextDataset(
        dataset_df=train_df,
        X_max_length=config.X_max_length,
        Y_max_length=config.Y_max_length,
        X_vocab_size=config.X_vocab_size,
        Y_vocab_size=config.Y_vocab_size,
        X_padding_idx=config.X_padding_idx,
        Y_padding_idx=config.Y_padding_idx,
    )

    val_dataset = CustomTextDataset(
        dataset_df=val_df,
        X_max_length=config.X_max_length,
        Y_max_length=config.Y_max_length,
        X_vocab_size=config.X_vocab_size,
        Y_vocab_size=config.Y_vocab_size,
        X_padding_idx=config.X_padding_idx,
        Y_padding_idx=config.Y_padding_idx,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        # num_workers=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        # num_workers=2,
    )

    lit_model = Seq2SeqModel(config=config)
    wandb_logger = WandbLogger(
        project=config.wandb_project,
        name=config.wandb_entity,
        log_model=False,
        config=config,
    )
    # lr_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = Trainer(
        max_epochs=config.epoch,
        accelerator="auto",
        log_every_n_steps=None,
        logger=wandb_logger,
    )  # Added accelerator gpu, can be cpu also, devices set to 1

    try:
        trainer.fit(lit_model, train_loader, val_loader)
    finally:
        # Mandatory cleanup
        wandb.finish()
        del lit_model, trainer, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()


config = Config()
## initialize the HPT
# sweep_id = wandb.sweep(sweep=sweep_configuration, project=config.wandb_project)
sweep_id = "urt26tw3"
wandb.agent(sweep_id, function=main, count=5, project=config.wandb_project)
