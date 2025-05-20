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
from lightning.pytorch.callbacks import ModelCheckpoint


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
    if os.path.exists(
        os.path.join("/kaggle", "input", "intro-to-da-a3-d-v2", "lexicons")
    ):
        DATASET_PATH = os.path.join(
            "/kaggle", "input", "intro-to-da-a3-d-v2", "lexicons"
        )
    else:
        DATASET_PATH = os.path.join("/kaggle", "input", "lexicons")

if args.colab:
    DATASET_PATH = os.path.join(
        "/content", "DA6401_Assignment_3", "dataset", "lexicons"
    )


train_df = pd.read_csv(os.path.join(DATASET_PATH, "ta.translit.sampled.train.idx.csv"))
val_df = pd.read_csv(os.path.join(DATASET_PATH, "ta.translit.sampled.dev.idx.csv"))


def main():
    """
    The main function that has all the code to run a training.
    This will be used by the sweep agent to run multiple hyperparamter
    tuning.
    """
    config = Config()

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

    ## Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.dirpath,
        filename=config.filename,
        monitor="val_word_acc_epoch",
        mode="max",
    )

    wandb_logger = WandbLogger(
        project=config.wandb_project,
        name=config.wandb_entity,
        log_model="all",
        config=config,
    )
    # lr_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = Trainer(
        max_epochs=config.epoch,
        accelerator="auto",
        log_every_n_steps=None,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )  # Added accelerator gpu, can be cpu also, devices set to 1

    trainer.fit(lit_model, train_loader, val_loader)


config = Config()

if __name__ == "__main__":
    main()
