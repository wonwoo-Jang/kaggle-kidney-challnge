import pandas as pd
import numpy as np
from PIL import Image
import torch
import os

from KidneyDataset import KidneyDataset
from torch.utils.data import Dataset, DataLoader

train_ds = KidneyDataset(
      val_stride=10,
      isValSet=False
    )

train_dl = DataLoader(
      train_ds,
      batch_size=32,
      num_workers=8,
      pin_memory=False
    )

if __name__ == '__main__':
  labels = Image.open('./dataset/train/kidney_1_dense/labels/0064.tif')
  labels = np.array(labels)
  # print(np.unique(labels))
  for _, batch_t in enumerate(train_dl):
    id, img, label, rle = batch_t
    print(id[0], torch.unique(label), label[0][0])
    if _>2:
      break