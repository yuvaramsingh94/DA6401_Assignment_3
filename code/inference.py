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
import json
import torch.nn.functional as F
from tqdm import tqdm
from utils import decoder_function

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
TEST_BATCH_SIZE = 2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


DATASET_PATH = os.path.join("dataset", "dakshina_dataset_v1.0", "ta", "lexicons")
if args.kaggle:
    if os.path.exists(os.path.join("/kaggle", "input", "intro-to-da-a3-d", "lexicons")):
        DATASET_PATH = os.path.join("/kaggle", "input", "intro-to-da-a3-d", "lexicons")
    else:
        DATASET_PATH = os.path.join("/kaggle", "input", "lexicons")

if args.colab:
    DATASET_PATH = os.path.join(
        "/content", "DA6401_Assignment_3", "dataset", "lexicons"
    )

test_df = pd.read_csv(os.path.join(DATASET_PATH, "ta.translit.sampled.test.idx.csv"))
train_df = pd.read_csv(os.path.join(DATASET_PATH, "ta.translit.sampled.train.idx.csv"))
val_df = pd.read_csv(os.path.join(DATASET_PATH, "ta.translit.sampled.dev.idx.csv"))

with open(
    os.path.join(DATASET_PATH, "tamil_token_index.json"), "r", encoding="utf-8"
) as f:
    tamil_idx = json.load(f)
tamil_idx_to_char = {j: i for i, j in tamil_idx.items()}

with open(
    os.path.join(DATASET_PATH, "english_token_index.json"), "r", encoding="utf-8"
) as f:
    english_idx = json.load(f)
english_idx_to_char = {j: i for i, j in english_idx.items()}

english_idx_to_char[26] = "-"
tamil_idx_to_char[48] = "-"


config = Config()  # is it

test_dataset = CustomTextDataset(
    dataset_df=train_df,
    X_max_length=config.X_max_length,
    Y_max_length=config.Y_max_length,
    X_vocab_size=config.X_vocab_size,
    Y_vocab_size=config.Y_vocab_size,
    X_padding_idx=config.X_padding_idx,
    Y_padding_idx=config.Y_padding_idx,
)


## TODO: load the trained weights
lit_model = Seq2SeqModel.load_from_checkpoint(
    checkpoint_path=config.WT_PATH,
    config=config,
)
lit_model = lit_model.eval()

## batch size = 1 for now
## Iterate only for the actual decoder length. this will reduce the computation
test_correct_prediction_count = 0
## TODO For loop over the test data. individual samples
prediction_result_dict = {"Input": [], "Actual_Y": [], "Prediction": []}
for idx in tqdm(range(len(test_dataset))):
    X, Y_decoder_ip, Y_decoder_op, X_len, Y_decoder_ip_len, Y_decoder_op_len = (
        test_dataset.__getitem__(idx)
    )
    X = torch.unsqueeze(X, 0).to(device)
    X_len = torch.unsqueeze(X_len, 0)  # .to(device)
    Y_decoder_ip = torch.unsqueeze(Y_decoder_ip, 0).to(device)
    prediction_list = []
    for i in range(1, Y_decoder_op_len + 1):  ## get the len of the decoder input
        new_decoder_ip = torch.unsqueeze(
            torch.cat(
                [
                    Y_decoder_ip[0][:i],
                    torch.tensor([config.Y_padding_idx] * (config.Y_max_length - i)).to(
                        device
                    ),
                ]
            ),
            axis=0,
        )

        ## Prediction
        (logits, attn_weight_list) = lit_model(X, X_len, new_decoder_ip)
        logits2 = logits.view(-1, logits.size(-1))
        prob = F.softmax(logits2, dim=1)
        preds = torch.argmax(prob, dim=1)
        prediction_list.append(torch.unsqueeze(preds[i], dim=0))

    prediction_tensor = torch.cat(prediction_list).to(device)
    correct = prediction_tensor == Y_decoder_op[:Y_decoder_op_len]

    ## Store the results to csv
    pred_str = decoder_function(
        ",".join([str(pc) for pc in prediction_tensor.detach().numpy()]),
        idx_to_char_dict=tamil_idx_to_char,
    )
    X_str = decoder_function(
        ",".join([str(xc) for xc in X[0].detach().numpy()]),
        idx_to_char_dict=english_idx_to_char,
    )
    actual_y_str = decoder_function(
        ",".join([str(ac) for ac in Y_decoder_op[:Y_decoder_op_len].detach().numpy()]),
        idx_to_char_dict=tamil_idx_to_char,
    )
    prediction_result_dict["Input"].append(X_str)
    prediction_result_dict["Actual_Y"].append(actual_y_str)
    prediction_result_dict["Prediction"].append(pred_str)

    if torch.all(correct):
        test_correct_prediction_count += 1
    else:
        test_correct_prediction_count += 0

test_accuracy = test_correct_prediction_count / len(test_dataset)
print("Test accuracy", test_accuracy)


pd.DataFrame.from_dict(prediction_result_dict).to_csv(config.PRED_CSV, index=False)
