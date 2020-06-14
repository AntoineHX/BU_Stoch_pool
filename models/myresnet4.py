'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from .stochsim import savg_pool2d
from .stoch import *



class SAvg_Pool2d(nn.Module):
    def __init__(self, stride=1, padding=0, dilation=1, groups=1,ceil_mode=True,bias=False,mode='s'):    
        super(SAvg_Pool2d, self).__init__()
        self.stride = stride
        self.mode = mode
        self.ceil_mode = ceil_mode

    def forward(self, x,stoch = True):
        out = savg_pool2d(x, self.stride, mode = self.mode,ceil_mode = self.ceil_mode)
        return out

stochmode = 'sim'#'sim'#'stride''stoch'''
finalstochpool = True
simmode = 'sbc'


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,pool=1):
        super(BasicBlock, self).__init__()
        
        if stochmode=='' or stride==1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        elif stochmode=='stride':
            if finalstochpool:
                stride = stride*pool
            self.conv1 = SConv2dStride(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        elif stochmode=='sim':  
            if finalstochpool:
                stride = stride*pool         
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=1, bias=False),
                SAvg_Pool2d(stride, mode = simmode,ceil_mode = True)
            )
        elif stochmode=='stoch':
            if finalstochpool:
                stride = stride*pool
            self.conv1 = SConv2dAvg(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if stochmode=='stoch':
            if pool!=1 and finalstochpool:
                self.conv2 = SConv2dAvg(planes, planes, kernel_size=3,
                                   stride=pool, padding=1, bias=False)
            else:
                self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=False)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if stochmode=='stride':
                self.shortcut = nn.Sequential(
                SConv2dStride(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
                )
            elif stochmode=='stoch':          
                self.shortcut = nn.Sequential(
                SConv2dAvg(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
                )
            elif stochmode=='sim':           
                self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=1, bias=False),
                SAvg_Pool2d(stride, mode = simmode,ceil_mode = True),
                nn.BatchNorm2d(self.expansion*planes)
                )
            elif stochmode=='':
                self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

#only basic block has been updated!!!
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        #self.conv1 = SConv2dStride(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SConv2dStride(planes, planes, kernel_size=3,stride=stride, padding=1, bias=False)
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = SConv2dStride(planes, self.expansion*planes, kernel_size=1, bias=False)
        #self.conv3 = nn.Conv2d(planes, self.expansion *                         planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                #nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                SConv2dStride(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
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
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,pool=4)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.stoch = stoch

    def _make_layer(self, block, planes, num_blocks, stride, pool=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,pool))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x ,stoch = True):
        #if self.training==False:
        #    stoch=False
        #stoch=True
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #if self.stoch:
        if stochmode=='':
            if not(finalstochpool):
            #if stochmode == '':
                out = F.avg_pool2d(out, 4)
            else:
                out = savg_pool2d(out, 4, mode = simmode)
        else:
            if not(finalstochpool):
                out = F.avg_pool2d(out, 4)
#        else:
#            if stoch:
#                out = savg_pool2d(out, 4, mode = 's')
#            else:
#                out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



def MyResNet18(stoch=False):
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
