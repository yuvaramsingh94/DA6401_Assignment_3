from torchvision.transforms import v2
from torchvision.transforms.v2 import Normalize
from torchvision.transforms import Resize
from torch.utils.data import Dataset
import pandas as pd
from utils import str_idx_to_list


class CustomTextDataset(Dataset):
    def __init__(
        self,
        dataset_df: pd.DataFrame,
    ):
        self.dataset_df = dataset_df

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, idx: int):
        X = str_idx_to_list(self.dataset_df.iloc[idx]["English"])  # .values[0]
        Y = str_idx_to_list(self.dataset_df.iloc[idx]["Tamil"])  # .values[0]
        ## X : English
        ## Y : Tamil
        return X, Y
