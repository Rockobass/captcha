import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.hub import load_state_dict_from_url


class CNN(nn.Module):
  def __init__(self, num_class=36, num_char=6):
    super(CNN, self).__init__()
    self.num_class = num_class
    self.num_char = num_char
    self.conv = nn.Sequential(
      # batch*3*200*50
      nn.Conv2d(3, 16, 3, padding=(1, 1)),
      nn.MaxPool2d(2, 2),
      nn.BatchNorm2d(16),
      nn.ReLU(),
      # batch*16*100*25
      nn.Conv2d(16, 64, 3, padding=(1, 1)),
      nn.MaxPool2d(2, 2),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      # batch*64*50*12
      nn.Conv2d(64, 512, 3, padding=(1, 1)),
      nn.MaxPool2d(2, 2),
      nn.BatchNorm2d(512),
      nn.ReLU(),
      # batch*512*25*6
      nn.Conv2d(512, 512, 3, padding=(1, 1)),
      nn.MaxPool2d(2, 2),
      nn.BatchNorm2d(512),
      nn.ReLU(),
      # batch*512*12*3
    )
    self.fc = nn.Linear(512 * 12 * 3, self.num_class * self.num_char)

  def forward(self, x):
    x = self.conv(x)

    x = x.view(-1, 512 * 12 * 3)

    # x = x.view(-1, 33792)
    x = self.fc(x)

    return x


class AlexNet(nn.Module):
  def __init__(self, num_class=36, num_char=6):  # imagenet数量
    super().__init__()
    self.num_class = num_class
    self.num_char = num_char
    self.layer1 = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
    )

    self.layer2 = nn.Sequential(
      nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, groups=2, padding=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
    )

    self.layer3 = nn.Sequential(
      nn.Conv2d(in_channels=256, out_channels=384, padding=1, kernel_size=3),
      nn.ReLU(inplace=True)
    )
    self.layer4 = nn.Sequential(
      nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
      nn.ReLU(inplace=True)
    )

    self.layer5 = nn.Sequential(
      nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2)
    )

    # 需要针对上一层改变view
    self.layer6 = nn.Sequential(
      nn.Linear(in_features=6 * 6 * 256, out_features=4096),
      nn.ReLU(inplace=True),
      nn.Dropout()
    )
    self.layer7 = nn.Sequential(
      nn.Linear(in_features=4096, out_features=4096),
      nn.ReLU(inplace=True),
      nn.Dropout()
    )

    self.layer8 = nn.Linear(in_features=4096, out_features=216)

  def forward(self, x):
    x = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
    x = x.view(-1, 6 * 6 * 256)
    x = self.layer7(self.layer6(x))
    x = self.layer8(self.layer7(self.layer6(x)))

    return x

class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, in_planes, planes, stride=1):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                             stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                             stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.shortcut = nn.Sequential()
    # 经过处理后的x要与x的维度相同(尺寸和深度)
    # 如果不相同，需要添加卷积+BN来变换为同一维度
    if stride != 1 or in_planes != self.expansion * planes:
       self.shortcut = nn.Sequential(
        nn.Conv2d(in_planes, self.expansion * planes,
                    kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(self.expansion * planes)
      )

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out += self.shortcut(x)
    out = F.relu(out)
    return out


# 用于ResNet50,101和152的残差块，用的是1x1+3x3+1x1的卷积
class Bottleneck(nn.Module):
  # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
  expansion = 4

  def __init__(self, in_planes, planes, stride=1):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                           stride=stride, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, self.expansion * planes,
                           kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(self.expansion * planes)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion * planes:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_planes, self.expansion * planes,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(self.expansion * planes)
      )

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = F.relu(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    out += self.shortcut(x)
    out = F.relu(out)
    return out


class ResNet(nn.Module):
  def __init__(self, block, num_blocks, num_classes=10):
    super(ResNet, self).__init__()
    self.in_planes = 64

    self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                           stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(64)

    self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
    self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
    self.linear = nn.Linear(512 * block.expansion, num_classes)

  def _make_layer(self, block, planes, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_planes, planes, stride))
      self.in_planes = planes * block.expansion
    return nn.Sequential(*layers)

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = F.avg_pool2d(out, 4)
    out = out.view(out.size(0), -1)
    out = self.linear(out)
    return out


def ResNet18():
  return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
  return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
  return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
  return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
  return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
  net = ResNet18()
  y = net(torch.randn(1, 3, 32, 32))
  print(y.size())


class FullyConvolutionalResnet18(models.ResNet):
  def __init__(self, num_classes=1000, pretrained=False, **kwargs):
    # Start with standard resnet18 defined here
    super().__init__(block=models.resnet.BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes, **kwargs)
    if pretrained:
      state_dict = load_state_dict_from_url(models.resnet.model_urls["resnet18"], progress=True)
      self.load_state_dict(state_dict)

    # Replace AdaptiveAvgPool2d with standard AvgPool2d
    self.avgpool = nn.AvgPool2d((7, 7))

    # Convert the original fc layer to a convolutional layer.
    self.last_conv = torch.nn.Conv2d(in_channels=self.fc.in_features, out_channels=num_classes, kernel_size=1)
    self.last_conv.weight.data.copy_(self.fc.weight.data.view(*self.fc.weight.data.shape, 1, 1))
    self.last_conv.bias.data.copy_(self.fc.bias.data)

    # Reimplementing forward pass.
  def _forward_impl(self, x):
    # Standard forward for resnet18
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)

    # Notice, there is no forward pass
    # through the original fully connected layer.
    # Instead, we forward pass through the last conv layer
    x = self.last_conv(x)
    return x

