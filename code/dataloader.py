from torch.utils.data import Dataset
import pandas as pd
from utils import str_idx_to_list
import torch


class CustomTextDataset(Dataset):
    def __init__(
        self,
        dataset_df: pd.DataFrame,
        X_max_length: int,
        Y_max_length: int,
        X_vocab_size: int,
        Y_vocab_size: int,
        X_padding_idx: int,
        Y_padding_idx: int,
    ) -> tuple[
        torch.tensor,
        torch.tensor,
        torch.tensor,
        torch.tensor,
        torch.tensor,
        torch.tensor,
    ]:
        """
        Text dataset for translit from English to Tamil task.

        Args:
            dataset_df (pd.DataFrame): _description_
            X_max_length (int): Maximum length of the word (in X) in the dataset
            Y_max_length (int): Maximum length of the word (in Y) in the dataset
            X_vocab_size (int): Size of the X vocabulary. This is to add padding integer.
            Y_vocab_size (int): Size of the Y vocabulary. This is to add padding integer.
        """
        self.dataset_df = dataset_df
        self.X_max_length = X_max_length
        self.Y_max_length = Y_max_length
        self.X_vocab_size = X_vocab_size
        self.Y_vocab_size = Y_vocab_size
        self.X_padding_idx = X_padding_idx
        self.Y_padding_idx = Y_padding_idx

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, idx: int):
        X = str_idx_to_list(self.dataset_df.iloc[idx]["English"])  # .values[0]
        Y = str_idx_to_list(self.dataset_df.iloc[idx]["Tamil"])  # .values[0]

        ## Decoder input y
        Y_decoder_ip = Y[:-1]
        ## Decoder output y
        Y_decoder_op = Y[1:]
        ## The actual length of the sequence
        X_len = len(X)
        Y_decoder_ip_len = len(Y_decoder_ip)
        Y_decoder_op_len = len(Y_decoder_op)
        if X_len < self.X_max_length:
            ## self.X_vocab_size refer to the padding index (last)
            X.extend([self.X_padding_idx] * (self.X_max_length - X_len))
        ## Decoder IP
        if Y_decoder_ip_len < self.Y_max_length:
            ## self.Y_vocab_size refer to the padding index (last)
            Y_decoder_ip.extend(
                [self.Y_padding_idx] * (self.Y_max_length - Y_decoder_ip_len)
            )
        if Y_decoder_op_len < self.Y_max_length:
            ## self.Y_vocab_size refer to the padding index (last)
            Y_decoder_op.extend(
                [self.Y_padding_idx] * (self.Y_max_length - Y_decoder_op_len)
            )
        ## Padding index
        ## X : English
        ## Y : Tamil

        X = torch.tensor(X, dtype=torch.long)
        Y_decoder_ip = torch.tensor(Y_decoder_ip, dtype=torch.long)
        Y_decoder_op = torch.tensor(Y_decoder_op, dtype=torch.long)

        X_len = torch.tensor(X_len, dtype=torch.long)
        Y_decoder_ip_len = torch.tensor(Y_decoder_ip_len, dtype=torch.int)
        Y_decoder_op_len = torch.tensor(Y_decoder_op_len, dtype=torch.int)
        return X, Y_decoder_ip, Y_decoder_op, X_len, Y_decoder_ip_len, Y_decoder_op_len
