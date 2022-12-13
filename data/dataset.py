import medmnist
import os
import torch

from medmnist import INFO
from torchvision import transforms as T

# Specify dataset
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DOWNLOAD_FLAG = True

# batch_size = 256

# n_channels = info['n_channels']
# n_classes = len(info['label'])

# print("n_classes", n_classes, info)

os.makedirs("./figs/", exist_ok=True)


class DataSet:
    def __init__(self):
        # Moves the range [0,1] to [-1,1]
        self.mean = torch.tensor([0.5], dtype=torch.float32)
        self.std = torch.tensor([0.5], dtype=torch.float32)

        self.plain_transform = T.Compose(
            [T.ToTensor(), T.Normalize(list(self.mean), list(self.std))]
        )

        DATA_FLAG = "pathmnist"
        info = INFO[DATA_FLAG]
        self.DataClass = getattr(medmnist, info["python_class"])

    def load_datasets(self):
        train_ds_plain = self.DataClass(
            split="train", transform=self.plain_transform, download=DOWNLOAD_FLAG
        )
        val_ds = self.DataClass(
            split="val", transform=self.plain_transform, download=DOWNLOAD_FLAG
        )
        test_ds = self.DataClass(
            split="test", transform=self.plain_transform, download=DOWNLOAD_FLAG
        )

        return train_ds_plain, val_ds, test_ds
