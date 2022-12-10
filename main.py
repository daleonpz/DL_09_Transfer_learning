import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import transforms as T

import matplotlib.pyplot as plt

import medmnist
from medmnist import INFO

from tqdm import tqdm

# Warning: local import - utils.py must be in the same folder as this notebook
from utils import *

# Specify dataset
data_flag   = 'pathmnist'
device      = 'cuda' if torch.cuda.is_available() else 'cpu'

download = True

batch_size = 256
info = INFO[data_flag]

n_channels = info['n_channels']
n_classes = len(info['label'])

print("n_classes", n_classes, info)
DataClass = getattr(medmnist, info['python_class'])

os.makedirs("./figs/", exist_ok=True)


# Moves the range [0,1] to [-1,1]
mean    = torch.tensor([0.5], dtype=torch.float32)
std     = torch.tensor([0.5], dtype=torch.float32)

plain_transform = T.Compose([T.ToTensor(), T.Normalize(list(mean), list(std))])

# load the data
train_ds_plain  = DataClass(split='train',  transform=plain_transform, download=download)
val_ds          = DataClass(split='val',    transform=plain_transform, download=download)
test_ds         = DataClass(split='test',   transform=plain_transform, download=download)

train_loader_plain1 = data.DataLoader(dataset=train_ds_plain, batch_size=batch_size, shuffle=True, drop_last=True)

img1, lab = next(iter(train_loader_plain1))

# show the images
plt.figure(figsize = (50,20))
for i in range(10):  
        imshow(train_ds_plain[i][0], i, mean, std)
