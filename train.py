import argparse
import pandas as pd
import numpy as np

import torch
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import KidneyModel
from KidneyDataset import KidneyDataset
from surfaceDiceMetric import score
from util import rle_batch_encode

class KidneyTraining:
  def __init__(self):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help="batch_size", default=32)
    parser.add_argument('--num_workers', type=int, help="cpu core: help background data loading", default=14)
    parser.add_argument('--epochs', type=int, help="epochs which you iterate", default=50)
    self.opt = parser.parse_args()

    self.use_cuda = torch.cuda.is_available()
    self.device = torch.device("cuda" if self.use_cuda else "cpu")

    self.model = self.initModel()
    self.optimizer = self.initOptimizer()
    self.writer = SummaryWriter()

  def initModel(self):
    model = KidneyModel(
            in_channels=1,
            n_classes=1,
            depth=3,
            wf=4,
            padding=True,
            batch_norm=True,
            up_mode='upconv',
        )
    if self.use_cuda:
      model.to(self.device)
    return model
    
  def initOptimizer(self):
    return Adam(self.model.parameters(), lr=0.01)
  
  def initTrainDL(self, val_stride):
    train_ds = KidneyDataset(
      val_stride=val_stride,
      isValSet=False
    )
    print("initTrainDL ready!\n")
    train_dl = DataLoader(
      train_ds,
      batch_size=self.opt.batch_size,
      num_workers=self.opt.num_workers,
      pin_memory=self.use_cuda,
      drop_last=True
    )

    return train_dl
  
  def initValDL(self, val_stride):
    val_ds = KidneyDataset(
      val_stride=val_stride,
      isValSet=True
    )

    val_dl = DataLoader(
      val_ds,
      batch_size=self.opt.batch_size,
      num_workers=self.opt.num_workers,
      pin_memory=self.use_cuda,
      drop_last=True
    )
    return val_dl

  def main(self):
    train_dl = self.initTrainDL(val_stride=10)
    val_dl = self.initValDL(val_stride=10)

    best_score = 0.0
    self.validation_cadence = 5
    for epoch_ndx in range(1, self.opt.epochs + 1):
        print("Epoch {} of {}, {}/{} batches of size {}*{}\n".format(
            epoch_ndx,
            self.opt.epochs,
            len(train_dl),
            len(val_dl),
            self.opt.batch_size,
            (torch.cuda.device_count() if self.use_cuda else 1),
        ))
        self.model.train()
        self.doTraining(epoch_ndx, train_dl)
        
        
        if epoch_ndx == 1 or epoch_ndx % self.validation_cadence == 0:
            # if validation is wanted
            self.model.eval()
            self.doValidation(epoch_ndx, val_dl)

  def doTraining(self, epoch_ndx, train_dl):
    for batch_ndx, batch_tup in enumerate(train_dl):
      self.optimizer.zero_grad()
      loss_var = self.computeBatchLoss(batch_ndx, batch_tup, train_dl.batch_size)
      self.writer.add_scalar("Loss/train", loss_var, epoch_ndx)
      loss_var.backward()

      print('Train Batch: {} / {}, dice_score: {}'.format(batch_ndx+1, len(train_dl), -1 * loss_var + 1))

      self.optimizer.step()

    return
  
  def doValidation(self, epoch_ndx, val_dl):
    for batch_ndx, batch_tup in enumerate(val_dl):
      
      loss_var = self.computeBatchLoss(batch_ndx+1, batch_tup, val_dl.batch_size)
      self.writer.add_scalar("Loss/valid", loss_var, epoch_ndx)
      print('Valid Batch: {} / {}, dice_score: {}'.format(batch_ndx+1, len(val_dl), -1 * loss_var + 1))

    return
  
  def dice_score(self, label_bool, pred_t, epsilon=0.001):
    
    dice_pred = pred_t.sum(dim=[1,2,3])
    dice_true = label_bool.sum(dim=[1,2,3])
    inter = (label_bool*pred_t).sum(dim=[1,2,3])
    
    dice = ((2*inter+epsilon)/(dice_true+dice_pred+epsilon))
    return dice

  def computeBatchLoss(self, batch_ndx:int, batch_tup, batch_size, threshold=0.5):
    id, img_t, label_t, rle_y = batch_tup

    img_t.requires_grad_(True)
    label_t.requires_grad_(True)
    

    img_t = img_t.to(self.device, non_blocking=True)
    label_t = label_t.to(self.device, non_blocking=True) 
    prediction_sig = self.model(img_t)
    label_bool = torch.where(label_t > threshold, 1.0, 0.0)
    dice_score = self.dice_score(label_bool, prediction_sig)

    return 1.0 - dice_score.mean()


if __name__ == '__main__':
  KidneyTraining().main()