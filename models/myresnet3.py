'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sconv2davg import SConv2dAvg

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, stoch=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.stoch=stoch
        if stoch: 
            self.conv2 = SConv2dAvg(planes, planes, kernel_size=3,
                               stride=1, padding=1) #bias=False) #Bias !?
        else :
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.stoch:
            _,_,h1,w1=out.shape
            h2,w2 = self.conv2.get_size(h1,w1)
            mask2 = torch.ones((h2,w2), device=x.device)
            selh2,selw2,mask1 = self.conv2.sample(h1,w1,mask=mask2)
            out = self.bn2(self.conv2(out,selh2,selw2,mask2,stoch=self.stoch))
        else:
            out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,stoch=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, stoch=stoch)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.stoch = stoch

        # if self.stoch:
        #     old_conv = self.layer4[-1].conv2
        #     self.layer4[-1].conv2=SConv2dAvg(old_conv.weight.shape[0], 
        #     old_conv.weight.shape[1], 
        #     old_conv.kernel_size, 
        #     stride=4)#old_conv.stride[0]) #Bias !?

    def _make_layer(self, block, planes, num_blocks, stride, stoch=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        if stoch:
            layers[-1]=block(self.in_planes, planes, stride, stoch=True)
        return nn.Sequential(*layers)

    def forward(self, x , stoch = False):
        #if self.training==False:
        #    stoch=False
        print(stoch)
        # self.layer1.stoch=stoch
        # self.layer2.stoch=stoch
        # self.layer3.stoch=stoch
        self.layer4[-1].stoch=stoch

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # print(out.shape)
        out = F.avg_pool2d(out, 4)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def MyResNet18(stoch=True):
    return ResNet(BasicBlock, [2, 2, 2, 2],stoch=stoch)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def MyResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
