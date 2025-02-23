# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as T
import os
import pandas as pd

def create_loader(data):
  df_labels = data['label'] 
  df_img = data.drop(['label'], 1)

  train_img_tensor = (torch.from_numpy(df_img.values).float() / 255).to(device)
  train_label_tensor = (torch.from_numpy(df_labels.values)).to(device)

  fc_dataset = T.TensorDataset(train_img_tensor, train_label_tensor)
  cnn_dataset = T.TensorDataset(train_img_tensor.view(-1, 1, 28, 28), train_label_tensor)
  
  fc_train_ds, fc_valid_ds = T.random_split(fc_dataset, [29400, 12600])
  cnn_train_ds, cnn_valid_ds = T.random_split(cnn_dataset, [29400, 12600])
  
  fc_train_dl = T.DataLoader(fc_train_ds, batch_size = 16)
  fc_valid_dl = T.DataLoader(fc_valid_ds, batch_size = 16)
  
  cnn_train_dl = T.DataLoader(cnn_train_ds, batch_size = 16)
  cnn_valid_dl = T.DataLoader(cnn_valid_ds, batch_size = 16)
  
  return (fc_train_dl, fc_valid_dl, cnn_train_dl, cnn_valid_dl)

def train_model(train_dl, valid_dl, net, criterion, optimizer):
  for i in range(5):
    print(f"\n================epoch {i+1}================\n")
    epoch(train_dl, valid_dl, net, criterion, optimizer)

def epoch(train_dl, valid_dl, net, criterion, optimizer):
  train_loss, train_acc = train_epoch(train_dl, net, criterion, optimizer)
  valid_loss, valid_acc = valid_epoch(valid_dl, net, criterion)
  print("==========Training==========")
  print(f"Loss: {train_loss}, Accuracy: {train_acc}")
  print("=========Validation=========")
  print(f"Loss: {valid_loss}, Accuracy: {valid_acc}")

def train_epoch(dl, net, criterion, optimizer):
  net.train()
  correct, total = 0, 0
  total_loss = 0
  for data in dl:
    input, label = data
    optimizer.zero_grad()
    output = net(input)
    loss = criterion(output, label)
    total_loss += loss.cpu().item()
    correct += cal_acc(output, label)
    total += len(label)
    loss.backward()
    optimizer.step()
  return (total_loss/total, correct/total)

def valid_epoch(dl, net, criterion):
  net.eval()
  correct, total = 0, 0
  total_loss = 0
  with torch.no_grad():
    for data in dl:
      input, label = data
      output = net(input)
      loss = criterion(output, label)
      total_loss += loss.cpu().item()
      correct += cal_acc(output, label)
      total += len(label)
  return (total_loss/total, correct/total) 

def cal_acc(pred, label):
  correct = 0
  for indx, i in enumerate(pred):
    if torch.argmax(i) == label[indx]:
      correct += 1
  return correct

class NNet(nn.Module):

  def __init__(self):
    super().__init__()
    self.fc = nn.Sequential(
      nn.Linear(784, 500),
      nn.BatchNorm1d(500),
      nn.ReLU(),
      nn.Linear(500, 500),
      nn.BatchNorm1d(500),
      nn.ReLU(),
      nn.Linear(500, 10)
    )

  def forward(self, x):
    x = self.fc(x)
    return x

class CNNet(nn.Module):

  def __init__(self):
    super().__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(1, 32, 5), #conv 1
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Conv2d(32, 64, 3),  #conv 2
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Conv2d(64, 128, 2), #conv 3
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.MaxPool2d(2)
    )

    self.fc = nn.Sequential(
      nn.Linear(128*2*2, 500), #fc 1
      nn.BatchNorm1d(500),
      nn.ReLU(),
      nn.Linear(500, 500), #fc 2
      nn.BatchNorm1d(500),
      nn.ReLU(),
      nn.Linear(500, 10) #out
    )

  def forward(self, x):
    x = self.conv(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x

if __name__ == "__main__":
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  os.chdir('drive/MyDrive/Digit Recognizer')

  train = pd.read_csv('train.csv')

  fc_train_dl, fc_valid_dl, cnn_train_dl, cnn_valid_dl = create_loader(train)

  fc_net = NNet().to(device)
  cnn_net = CNNet().to(device)
  criterion = nn.CrossEntropyLoss().to(device)
  fc_optimizer = optim.Adam(fc_net.parameters(), lr = 0.001)
  cnn_optimizer = optim.Adam(cnn_net.parameters(), lr = 0.001)

  print("======================Normal NN======================")
  train_model(fc_train_dl, fc_valid_dl, fc_net, criterion, fc_optimizer)
  torch.save(fc_net, "Models/fc_net.pth")
  print("\n\n======================CNN======================")
  train_model(cnn_train_dl, cnn_valid_dl, cnn_net, criterion, cnn_optimizer)
  torch.save(cnn_net, "Models/cnn_net.pth")
