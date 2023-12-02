from collections import namedtuple

import copy
from PIL import Image
import os
import numpy as np
import pandas as pd
import random

from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms

LabelsTuple = namedtuple(
  'LabelsTuple',
  'id, rle'
)

def getLabelsList():
  rles = pd.read_csv('./train_rles.csv')

  label_list = []
  for _, ser in rles.iterrows():
    id = ser['id']
    rle = ser['rle']
    label_list.append(LabelsTuple(
      id,
      rle
    ))
  label_list.sort()

  return label_list

class Kidney:
  def __init__(self, id):
    # seed = random.randint(1, 2000)
    # torch.manual_seed(seed)
    # crop_size = 256
    self.transforms = transforms.Compose([transforms.Resize(size=(512, 512), interpolation=Image.NEAREST)])

    folder, ind = ('_').join(id.split('_')[:-1]), id.split('_')[-1]
    img_train = Image.open(os.path.join('./dataset/train', folder, 'images', ind + '.tif'))
    img_label = Image.open(os.path.join('./dataset/train', folder, 'labels', ind + '.tif'))

    # img_train_crop = self.transforms(img_train)
    # img_label_crop = self.transforms(img_label)
    img = self.transforms(img_train)
    label = self.transforms(img_label)

    img_a = np.array(img, float)
    label_a = np.array(label, float)
    self.img_a = img_a
    self.label_a = label_a

def getKidney(id):
  return Kidney(id)

class KidneyDataset(Dataset):
  def __init__(self, val_stride:int=0, isValSet:bool=None, id:str=None):
    
    self.label_list = copy.copy(getLabelsList())
    if id:
      self.label_list = [i for i in self.label_list if i.id == id]
    
    if isValSet:
      assert val_stride > 0, val_stride
      self.label_list = self.label_list[::val_stride]
    elif val_stride:
      assert val_stride > 0, val_stride
      del self.label_list[::val_stride]

  def __len__(self):
    return len(self.label_list)
  
  def __getitem__(self, ind):
    id = self.label_list[ind].id
    rle = self.label_list[ind].rle
    kidney = getKidney(id)
    img_t = torch.from_numpy(kidney.img_a)
    img_t = img_t.to(torch.float32)
    img_t = img_t.unsqueeze(0)
    label_t = torch.from_numpy(kidney.label_a)
    label_t = label_t.to(torch.float32)
    label_t = label_t.unsqueeze(0)

    return (id, img_t, label_t, rle)