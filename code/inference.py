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

test_df = pd.read_csv(os.path.join(DATASET_PATH, "ta.translit.sampled.test.idx.csv"))
# train_df = pd.read_csv(os.path.join(DATASET_PATH, "ta.translit.sampled.train.idx.csv"))
# val_df = pd.read_csv(os.path.join(DATASET_PATH, "ta.translit.sampled.dev.idx.csv"))

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
    dataset_df=test_df,
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
prediction_result_dict = {
    "Input": [],
    "Actual_Y": [],
    "Prediction": [],
    "Actual_Y_idx": [],
    "Prediction_idx": [],
}
# predicted_words = []

for idx in tqdm(range(len(test_dataset))):
    X, _, Y_decoder_op, X_len, _, Y_decoder_op_len = test_dataset[idx]
    X = X.unsqueeze(0).to(device)
    X_len = X_len.unsqueeze(0)

    # Perform encoder on the input
    encoder_outputs, encoder_hidden = lit_model.encoder(X, X_len)

    ## Start token for the decoder
    decoder_input = torch.tensor([[0]], device=device)
    hidden = encoder_hidden
    decoded_indices = []

    for _ in range(config.Y_max_length):
        if config.attention_model:
            # decoder returns logits; some implementations also return new hidden
            logits, hidden, attn_weight_list = lit_model.decoder(
                decoder_input,
                encoder_outputs,
                hidden,
                (X != config.X_padding_idx).int(),
            )
            # If your attention decoder returns (logits, new_hidden), unpack accordingly
        else:
            logits, hidden = lit_model.decoder(decoder_input, hidden)

        # logits: (1, 1, V) → flatten to (V,)
        logits = logits.view(-1, logits.size(-1))
        next_token = torch.argmax(F.softmax(logits, dim=1), dim=1).item()

        # Stop at end‐of‐sequence
        if next_token == config.Y_padding_idx:
            break

        decoded_indices.append(next_token)
        # Prepare next input: shape (1,1)
        decoder_input = torch.tensor([[next_token]], device=device)

    # --- 5. Convert indices → string and store ---
    word = "".join(tamil_idx_to_char[i] for i in decoded_indices[:Y_decoder_op_len])
    # predicted_words.append(word)

    # Now predicted_words[i] holds the full predicted word for sample i
    # print(predicted_words[:5])
    actual_Y_tokens = Y_decoder_op[:Y_decoder_op_len].detach().cpu().numpy().tolist()
    correct = decoded_indices[:Y_decoder_op_len] == actual_Y_tokens

    X_str = decoder_function(
        ",".join([str(xc) for xc in X[0].detach().cpu().numpy()]),
        idx_to_char_dict=english_idx_to_char,
    )
    actual_y_str = decoder_function(
        ",".join(
            [str(ac) for ac in Y_decoder_op[:Y_decoder_op_len].detach().cpu().numpy()]
        ),
        idx_to_char_dict=tamil_idx_to_char,
    )
    prediction_result_dict["Input"].append(X_str)
    prediction_result_dict["Actual_Y"].append(actual_y_str)
    prediction_result_dict["Prediction"].append(word)

    prediction_result_dict["Actual_Y_idx"].append(
        Y_decoder_op[:Y_decoder_op_len].detach().cpu().numpy().tolist()
    )
    prediction_result_dict["Prediction_idx"].append(decoded_indices[:Y_decoder_op_len])

    if correct:
        test_correct_prediction_count += 1
    else:
        test_correct_prediction_count += 0

test_accuracy = test_correct_prediction_count / len(test_dataset)
print("Test accuracy", test_accuracy)


pd.DataFrame.from_dict(prediction_result_dict).to_csv(config.PRED_CSV, index=False)
