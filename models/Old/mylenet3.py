'''LeNet in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from .sconv2davg import SConv2dAvg

class MyLeNetNormal(nn.Module):#epoch 12s
    def __init__(self):
        super(MyLeNetNormal, self).__init__()
        self.conv1 = nn.Conv2d(3, 200, 5, stride=1)
        self.conv2 = nn.Conv2d(200, 400, 3, stride=1)
        self.conv3 = nn.Conv2d(400, 800, 3, stride=1)
        self.fc1   = nn.Linear(800, 10)

    def forward(self, x, stoch=True):
        _,_,h0,w0 = x.shape
        out = F.relu(self.conv1(x))
        _,_,h1,w1 = out.shape
        out = F.avg_pool2d(out,2,ceil_mode=True)
        out = F.relu(self.conv2(out))
        _,_,h2,w2 = out.shape
        out = F.avg_pool2d(out,2,ceil_mode=True)
        out = F.relu(self.conv3(out))
        out = F.avg_pool2d(out,4,ceil_mode=True)
        
        out = out.view(out.size(0), -1 )
        out = (self.fc1(out))

        return out

def savg_pool2d(x,size,ceil_mode=False):
    b,c,h,w = x.shape
    device = x.device
    if ceil_mode:
        out_h = math.ceil(h/size)
        out_w = math.ceil(w/size)
    else:
        out_h = math.floor(h/size)
        out_w = math.floor(w/size)
    selh = torch.randint(size,(out_h,out_w), device=device)
    #selh[:] = 0
    rngh = torch.arange(0,h,size,device=x.device).view(-1,1)
    selh = selh+rngh

    selw = torch.randint(size,(out_h,out_w), device=device)
    #selw[:] = 0
    rngw = torch.arange(0,w,size,device=x.device)
    selw = selw+rngw

    newx = x[:,:, selh, selw]
    return newx

def savg_pool2d_(x,size,ceil_mode=False):
    b,c,h,w = x.shape
    device = x.device
    selh = torch.randint(size,(math.floor(h/size),math.floor(w/size)), device=device)
    rngh = torch.arange(0,h,size, device=device).long().view(h/size,1).repeat(1,w/size).view(math.floor(h/size),math.floor(w/size))
    selx = (selh+rngh).repeat(b,c,1,1)

    selw = torch.randint(size,(math.floor(h/size),math.floor(w/size)), device=device)
    rngw = torch.arange(0,w,size, device=device).long().view(1,h/size).repeat(h/size,1).view(math.floor(h/size),math.floor(w/size))
    sely = (selw+rngw).repeat(b,c,1,1)
    bv, cv ,hv, wv = torch.meshgrid([torch.arange(0,b), torch.arange(0,c),torch.arange(0,h/size),torch.arange(0,w/size)])
    #x=x.view(b,c,h*w)
    newx = x[bv,cv, selx, sely]
    #ghdh
    return newx

class MyLeNetSimNormal(nn.Module):#epoch 12s
    def __init__(self):
        super(MyLeNetSimNormal, self).__init__()
        self.conv1 = nn.Conv2d(3, 200, 5, stride=1)
        self.conv2 = nn.Conv2d(200, 400, 3, stride=1)
        self.conv3 = nn.Conv2d(400, 800, 3, stride=1)
        self.fc1   = nn.Linear(800, 10)

    def forward(self, x, stoch=True):

        #stoch=True
        out = F.relu(self.conv1(x))
        if stoch:
            out = savg_pool2d(out,2,ceil_mode=True)
        else:
            out = F.avg_pool2d(out,2,ceil_mode=True)
        out = F.relu(self.conv2(out))
        if stoch:
            out = savg_pool2d(out,2,ceil_mode=True)
        else:
            out = F.avg_pool2d(out,2,ceil_mode=True)
        out = F.relu(self.conv3(out))
        if stoch:
            out = savg_pool2d(out,4,ceil_mode=True)
        else:
            out = F.avg_pool2d(out,4,ceil_mode=True)
        
        out = out.view(out.size(0), -1 )
        out = (self.fc1(out))
        return out


class MyLeNetStride(nn.Module):#epoch 6s
    def __init__(self):
        super(MyLeNetStride, self).__init__()
        self.conv1 = nn.Conv2d(3, 200, 5, stride=2)
        self.conv2 = nn.Conv2d(200, 400, 3, stride=2)
        self.conv3 = nn.Conv2d(400, 800, 3, stride=4)
        self.fc1   = nn.Linear(800, 10)

    def forward(self, x, stoch=True):

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))

        out = out.view(out.size(0), -1 )
        out = (self.fc1(out))
        return out

class MyLeNetMatNormal(nn.Module):#epach 21s
    def __init__(self):
        super(MyLeNetMatNormal, self).__init__()
        self.conv1 = SConv2dAvg(3, 200, 5, stride=1)
        self.conv2 = SConv2dAvg(200, 400, 3, stride=1)
        self.conv3 = SConv2dAvg(400, 800, 3, stride=1)
        self.fc1   = nn.Linear(800, 10)

    def forward(self, x, stoch=True):
        _,_,h0,w0 = x.shape
        out = F.relu(self.conv1(x))
        out = F.avg_pool2d(out,2,ceil_mode=True)

        _,_,h1,w1 = out.shape
        out = F.relu(self.conv2(out))
        out = F.avg_pool2d(out,2,ceil_mode=True)

        _,_,h2,w2 = out.shape
        out = F.relu(self.conv3(out))
        out = F.avg_pool2d(out,4,ceil_mode=True)

        out = out.view(out.size(0), -1 )
        out = (self.fc1(out))

        if 1:
            comp = 0
            comp+=self.conv1.comp(h0,w0)
            comp+=self.conv2.comp(h1,w1)
            comp+=self.conv3.comp(h2,w2)
            self.comp = comp/1000000
        return out


class MyLeNetMatStoch(nn.Module):#epoch 17s
    def __init__(self):
        super(MyLeNetMatStoch, self).__init__()
        self.conv1 = SConv2dAvg(3, 200, 5, stride=2,ceil_mode=True)
        self.conv2 = SConv2dAvg(200, 400, 3, stride=2,ceil_mode=True)
        self.conv3 = SConv2dAvg(400, 800, 3, stride=4,ceil_mode=True)
        self.fc1   = nn.Linear(800, 10)

    def forward(self, x, stoch=True):
        # if stoch:
        _,_,h0,w0=x.shape
        out = F.relu(self.conv1(x,stoch=stoch))
        _,_,h1,w1=out.shape
        out = F.relu(self.conv2(out,stoch=stoch))
        _,_,h2,w2=out.shape
        out = F.relu(self.conv3(out,stoch=stoch))
        # else:
        #     out = F.relu(self.conv1(x,stoch=True,stride=1))
        #     out = F.avg_pool2d(out,2,ceil_mode=True)
        #     out = F.relu(self.conv2(out,stoch=True,stride=1))
        #     out = F.avg_pool2d(out,2,ceil_mode=True)
        #     out = F.relu(self.conv3(out,stoch=True,stride=1))
        #     out = F.avg_pool2d(out,4,ceil_mode=True)

        out = out.view(out.size(0), -1 )
        out = self.fc1(out)
        #Estimate computation
        if 1:
            comp = 0
            comp+=self.conv1.comp(h0,w0)
            comp+=self.conv2.comp(h1,w1)
            comp+=self.conv3.comp(h2,w2)
            self.comp = comp/1000000
        return out
    
class MyLeNetMatStochBU(nn.Module):#epoch 11s 
    def __init__(self):
        super(MyLeNetMatStochBU, self).__init__()
        self.conv1 = SConv2dAvg(3, 200, 5, stride=2,ceil_mode=True)
        self.conv2 = SConv2dAvg(200, 400, 3, stride=2,ceil_mode=True)
        self.conv3 = SConv2dAvg(400, 800, 3, stride=4,ceil_mode=True)
        self.fc1   = nn.Linear(800, 10)

    def forward(self, x, stoch=True):
        #get sizes
        h0,w0 = x.shape[2],x.shape[3]
        h1,w1 = self.conv1.get_size(h0,w0)
        h2,w2 = self.conv2.get_size(h1,w1)
        h3,w3 = self.conv3.get_size(h2,w2)
        # print('Shapes :')
        # print('0', h0, w0)
        # print('1', h1, w1)
        # print('2', h2, w2)
        # print('3', h3, w3)
        #sample BU
        # mask3 = torch.ones(h3,w3).cuda()
        mask3 = torch.ones((h3,w3), device=x.device)
        selh3,selw3,mask2 = self.conv3.sample(h2,w2,mask=mask3)
        selh2,selw2,mask1 = self.conv2.sample(h1,w1,mask=mask2)
        selh1,selw1,mask0 = self.conv1.sample(h0,w0,mask=mask1)
        #forward
        if stoch:
            out = F.relu(self.conv1(x,selh1,selw1,mask1,stoch=stoch))
            out = F.relu(self.conv2(out,selh2,selw2,mask2,stoch=stoch))
            out = F.relu(self.conv3(out,selh3,selw3,mask3,stoch=stoch))
        else:
            out = F.relu(self.conv1(x,stoch=True,stride=1))
            out = F.avg_pool2d(out,2,ceil_mode=True)
            out = F.relu(self.conv2(out,stoch=True,stride=1))
            out = F.avg_pool2d(out,2,ceil_mode=True)
            out = F.relu(self.conv3(out,stoch=True,stride=1))
            out = F.avg_pool2d(out,4,ceil_mode=True)

        out = out.view(out.size(0), -1 )
        out = (self.fc1(out))
        #Estimate computation
        if 1:
            comp = 0
            comp+=self.conv1.comp(h0,w0,mask1)
            comp+=self.conv2.comp(h1,w1,mask2)
            comp+=self.conv3.comp(h2,w2,mask3)
            self.comp = comp.item()/1000000
        return out

