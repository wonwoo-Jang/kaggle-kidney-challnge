from PIL import Image
import numpy as np
import pandas as pd

from model import KidneyModel
from util import rle_encode

import torch
from torch.optim import SGD, Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate
from torchvision import transforms

import argparse
import os

class kidneyTestDataset(Dataset):
  def __init__(self, data_root):
    testIdList=[]
    self.data_root = data_root
    for dir in os.listdir(data_root):
      for file in os.listdir(os.path.join(data_root, dir, 'images')):
        testIdList.append(dir + '_' + file[:-4])
    self.testIdList = testIdList
    self.transforms = transforms.Compose([transforms.Resize(size=(512, 512), interpolation=Image.NEAREST)])
  
  def __len__(self):
    return len(self.testIdList)
  
  def __getitem__(self, ind):
    id = self.testIdList[ind]
    folder, ind = ('_').join(id.split('_')[:-1]), id.split('_')[-1]
    path = os.path.join(self.data_root, folder, 'images', ind+'.tif')
    img_test = Image.open(path)
    size = img_test.size
    img = self.transforms(img_test)
    img_a = np.array(img, float)
    img_t = torch.from_numpy(img_a)
    img_t = img_t.to(torch.float32)
    img_t = img_t.unsqueeze(0)
    return (id, img_t, size)

class kidneyTest:
  def __init__(self):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help="batch_size for inference", default=1)
    parser.add_argument('--data_root', type=str, help="test data root", default='./dataset/test')
    parser.add_argument('--num_workers', type=int, help="cpu core: help background data loading", default=14)
    parser.add_argument('--checkpoint', type=str, help="checkpoint root", default='./checkpoints/3ch.pth')
    self.use_cuda = torch.cuda.is_available()
    self.device = torch.device("cuda" if self.use_cuda else "cpu")

    self.opt = parser.parse_args()
    self.model = KidneyModel(
            in_channels=1,
            n_classes=1,
            depth=3,
            wf=4,
            padding=True,
            batch_norm=True,
            up_mode='upconv',
        )
    self.optimizer = Adam(self.model.parameters())
    self.model.load_state_dict(torch.load(self.opt.checkpoint))
    self.model.to(self.device)
    self.model.eval()

  def initTestDL(self):
    test_ds = kidneyTestDataset(
      self.opt.data_root
    )

    test_dl = DataLoader(
      test_ds,
      batch_size=self.opt.batch_size,
      num_workers=self.opt.num_workers,
      pin_memory=self.use_cuda,
    )

    return test_dl
    
    
  def main(self):
    result = pd.DataFrame({'id':[],'rle':[]})

    test_dl = self.initTestDL()
    with torch.no_grad():
      for _, batch in enumerate(test_dl):
        id, img_t, size = batch

        img_t = img_t.to(self.device, non_blocking=True)
        pred_sig = self.model(img_t)
        pred_sig = interpolate(pred_sig, size=size, mode='bilinear')
        pred_bool = torch.where(pred_sig > 0.5, 1, 0)
        for i in range(self.opt.batch_size):
          target_t = pred_bool[i].squeeze()
          rle = rle_encode(target_t.cpu().numpy())
          new_result = pd.DataFrame({'id': [id[i]],
                                    'rle': [rle]})
          result = pd.concat([result, new_result])
    result.to_csv('submission.csv', index=False)

if __name__ == '__main__':
  kidneyTest().main()