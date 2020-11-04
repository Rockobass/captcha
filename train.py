import torch
import torch.nn as nn
from torch.autograd import Variable
from models import CNN, ResNet18
from datasets import CaptchaData
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop
from torchvision.models import resnet18

import time
import os

batch_size = 250
base_lr = 0.001
max_epoch = 120
model_path = './checkpoints/model.pth'
restor = False

if not os.path.exists('./checkpoints'):
  os.mkdir('./checkpoints')


def calculat_acc(output, target):
  output, target = output.view(-1, 36), target.view(-1, 36)
  output = nn.functional.softmax(output, dim=1)
  output = torch.argmax(output, dim=1)
  target = torch.argmax(target, dim=1)
  output, target = output.view(-1, 6), target.view(-1, 6)
  correct_list = []
  for i, j in zip(target, output):
    if torch.equal(i, j):
      correct_list.append(1)
    else:
      correct_list.append(0)
  acc = sum(correct_list) / len(correct_list)
  return acc


def train():
  transforms = Compose([Resize((200, 50)), ToTensor()])
  train_dataset = CaptchaData('E:/pycharm_demos/captcha/data/train', transform=transforms)
  train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0,
                                 shuffle=True, drop_last=True)
  test_data = CaptchaData('E:/pycharm_demos/captcha/data/test', transform=transforms)
  test_data_loader = DataLoader(test_data, batch_size=batch_size,
                                num_workers=0, shuffle=True, drop_last=True)
  cnn = CNN()
  if torch.cuda.is_available():
    print("gpu加速啦")
    cnn.cuda()
  if restor:
    cnn.load_state_dict(torch.load(model_path))
  #        freezing_layers = list(cnn.named_parameters())[:10]
  #        for param in freezing_layers:
  #            param[1].requires_grad = False
  #            print('freezing layer:', param[0])

  optimizer = torch.optim.Adam(cnn.parameters(), lr=base_lr)
  criterion = nn.MultiLabelSoftMarginLoss()

  for epoch in range(max_epoch):
    start_ = time.time()

    loss_history = []
    acc_history = []
    cnn.train()
    for img, target in train_data_loader:
      img = Variable(img)
      target = Variable(target)
      if torch.cuda.is_available():
        img = img.cuda()
        target = target.cuda()
      output = cnn(img)
      loss = criterion(output, target)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      acc = calculat_acc(output, target)
      acc_history.append(float(acc))
      loss_history.append(float(loss))
    print('train_loss: {:.4}|train_acc: {:.4}'.format(
      torch.mean(torch.Tensor(loss_history)),
      torch.mean(torch.Tensor(acc_history)),
    ))

    loss_history = []
    acc_history = []
    cnn.eval()
    for img, target in test_data_loader:
      img = Variable(img)
      target = Variable(target)
      if torch.cuda.is_available():
        img = img.cuda()
        target = target.cuda()
      output = cnn(img)
      loss = criterion(output, target)
      acc = calculat_acc(output, target)
      acc_history.append(float(acc))
      loss_history.append(float(loss))

    print('test_loss: {:.4}|test_acc: {:.4}'.format(
      torch.mean(torch.Tensor(loss_history)),
      torch.mean(torch.Tensor(acc_history)),
    ))
    print('epoch: {}|time: {:.4f}'.format(epoch, time.time() - start_))
    torch.save(cnn.state_dict(), model_path)


if __name__ == "__main__":
  train()
  pass
