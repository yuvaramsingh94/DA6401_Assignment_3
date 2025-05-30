from Config import Config
from torch.utils.data import DataLoader
import torch
import wandb
import gc
import numpy as np
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
from utils import decoder_function, color_code_text

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


wandb.init(
    project=config.wandb_project,
    name="prediction_basic_v5",
    config=config,
)


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
    "Sample_id": [],
    "Input": [],
    "Actual_Y": [],
    "Prediction": [],
    "Actual_Y_idx": [],
    "Prediction_idx": [],
    "Correct": [],
}
# predicted_words = []
top_9_attention_maps_list = []
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
    attention_map_list = []
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
            attn_weight_list = attn_weight_list.squeeze(0)
            attention_map_list.append(attn_weight_list)
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
    if config.attention_model:
        attention_map = (
            torch.concat(attention_map_list, dim=0)
            .detach()
            .cpu()
            .numpy()[:Y_decoder_op_len, :X_len]
        )
        np.savez(
            os.path.join(config.attn_map_path, f"{idx}.npz"),
            attention_map=attention_map,
        )

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
    prediction_result_dict["Sample_id"].append(idx)
    prediction_result_dict["Correct"].append(correct)
    prediction_result_dict["Input"].append(X_str.replace("-", ""))
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
wandb.log({"Test accuracy": test_accuracy})

pred_df = pd.DataFrame.from_dict(prediction_result_dict)
pred_df["Correct"] = pred_df["Correct"] * 1
pred_df.to_csv(config.PRED_CSV, index=False)

## Load the prediction df
# pred_df = pd.read_csv(os.path.join("predictions_vanilla", "best_basic_prediction.csv"))
## Get some correct and wrongly predicted samples
pred_df = pred_df.sample(frac=1, random_state=SEED)
pred_to_print_df = pd.concat(
    [pred_df.query("Correct == 1")[:5], pred_df.query("Correct == 0")[:5]]
)


pred_to_print_df["colored prediction"] = pred_to_print_df.apply(color_code_text, axis=1)

# Generate HTML table without escaping HTML tags
html_table = pred_to_print_df[["Input", "Actual_Y", "colored prediction"]].to_html(
    escape=False, index=False
)

html_content = f"""<!DOCTYPE html>
<html lang="ta">
<head>
  <meta charset="utf-8">
  <title>Prediction table</title>
  <style>
    body {{
      font-family: 'Noto Sans Tamil', 'Lohit Tamil', sans-serif;
    }}
    table.dataframe {{
      border-collapse: collapse;
      margin: 1em;
    }}
    th, td {{
      border: 1px solid #666;
      padding: 4px 8px;
      text-align: left;
      vertical-align: top;
    }}
  </style>
</head>
<body>
  {html_table}
</body>
</html>"""

wandb.log({"Prediction_table": wandb.Html(html_content)})


## Wandb log the confusion matrix
Y_true = []
for i in pred_df["Actual_Y_idx"]:
    Y_true.extend(i)
Y_pred = []
for i in pred_df["Prediction_idx"]:
    Y_pred.extend(i)

tamil_idx_to_char = dict(sorted(tamil_idx_to_char.items()))
class_label = list(tamil_idx_to_char.values())


wandb.log(
    {
        "Test_confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            preds=Y_pred,
            y_true=Y_true,
            class_names=class_label,
        )
    }
)
