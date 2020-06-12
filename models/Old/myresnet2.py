'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
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
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3,
                               stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.stoch = stoch

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def myconv2d_avg(self, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1,size=2):
        batch_size, in_channels, in_h, in_w = input.shape
        out_channels, in_channels, kh, kw =  weight.shape
        out_h = (in_h+2*padding)-2*(int(kh)/2)
        out_w = (in_w+2*padding)-2*(int(kw)/2)

        unfold = torch.nn.Unfold(kernel_size=(kh, kw), dilation=dilation, padding=padding, stride=stride)
        inp_unf = unfold(input).view(batch_size,in_channels*kh*kw,out_h,out_w)
        sel_h = torch.LongTensor(out_h/size,out_w/size).random_(0, size)#.cuda()
        rng_h = sel_h + torch.arange(0,out_h,size).long()#.cuda()

        sel_w = torch.LongTensor(out_h/size,out_w/size).random_(0, size)#.cuda()
        rng_w = sel_w+torch.arange(0,out_w,size).long()#.cuda()
        inp_unf = inp_unf[:,:,rng_h,rng_w].view(batch_size,in_channels*kh*kw,out_h/size*out_w/size)
        #unfold_avg = torch.nn.Unfold(kernel_size=(1, 1), dilation=1, padding=0, stride=2)

        if bias is None:
            out_unf = inp_unf.transpose(1, 2).matmul(weight.view(weight.size(0), -1).t()).transpose(1, 2)
        else:
            out_unf = (inp_unf.transpose(1, 2).matmul(weight.view(weight.size(0), -1).t()) + bias).transpose(1, 2)

        out = out_unf.view(batch_size, out_channels, out_h/size, out_w/size).contiguous()
        return out


    def savg_pool2d(self,x,size,locx=-1,locy=-1):
        b,c,h,w = x.shape
        if locx==-1:
            selh = torch.LongTensor(h/size,w/size).random_(0, size)
        else:
            selh = torch.ones(h/size,w/size).long()*loc
        rngh = torch.arange(0,h,size).long().view(h/size,1).repeat(1,w/size).view(h/size,w/size)
        selx = (selh+rngh).repeat(b,c,1,1)
        if locy==-1:
            selw = torch.LongTensor(h/size,w/size).random_(0, size)
        else:
            selw = torch.ones(h/size,w/size).long()*loc
        rngw = torch.arange(0,w,size).long().view(1,h/size).repeat(h/size,1).view(h/size,w/size)
        sely = (selw+rngw).repeat(b,c,1,1)
        bv, cv ,hv, wv = torch.meshgrid([torch.arange(0,b), torch.arange(0,c),torch.arange(0,h/size),torch.arange(0,w/size)])
        #x=x.view(b,c,h*w)
        newx = x[bv,cv, selx, sely]
        #ghdh
        return newx

    def forward(self, x ,stoch = True):
        #if self.training==False:
        #    stoch=False
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if self.stoch and stoch:
            out = F.relu(self.myconv2d_avg(out, self.conv2.weight, bias=self.conv2.bias,padding=1,size=4)) 
            #out = F.avg_pool2d(out, 2)
        else:
            out = F.relu(self.myconv2d_avg(out, self.conv2.weight, bias=self.conv2.bias,padding=1,size=1))
            out = F.avg_pool2d(out, 4)
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
