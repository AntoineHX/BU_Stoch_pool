'''LeNet in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyLeNet(nn.Module):
    def __init__(self):
        super(MyLeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)        

    def savg_pool2d(self,x,size):
        b,c,h,w = x.shape
        selh = torch.LongTensor(h/size,w/size).random_(0, size)
        rngh = torch.arange(0,h,size).long().view(h/size,1).repeat(1,w/size).view(h/size,w/size)
        selx = (selh+rngh).repeat(b,c,1,1)

        selw = torch.LongTensor(h/size,w/size).random_(0, size)
        rngw = torch.arange(0,w,size).long().view(1,h/size).repeat(h/size,1).view(h/size,w/size)
        sely = (selw+rngw).repeat(b,c,1,1)
        bv, cv ,hv, wv = torch.meshgrid([torch.arange(0,b), torch.arange(0,c),torch.arange(0,h/size),torch.arange(0,w/size)])
        #x=x.view(b,c,h*w)
        newx = x[bv,cv, selx, sely]
        #ghdh
        return newx

    def ssoftmax_pool2d(self,x,size,idx):
        b,c,h,w = x.shape
        w = wdataset[idx]
        selh = torch.LongTensor(h/size,w/size).random_(0, size)
        rngh = torch.arange(0,h,size).long().view(h/size,1).repeat(1,w/size).view(h/size,w/size)
        selx = (selh+rngh).repeat(b,c,1,1)

        selw = torch.LongTensor(h/size,w/size).random_(0, size)
        rngw = torch.arange(0,w,size).long().view(1,h/size).repeat(h/size,1).view(h/size,w/size)
        sely = (selw+rngw).repeat(b,c,1,1)
        bv, cv ,hv, wv = torch.meshgrid([torch.arange(0,b), torch.arange(0,c),torch.arange(0,h/size),torch.arange(0,w/size)])
        #x=x.view(b,c,h*w)
        newx = x[bv,cv, selx, sely]
        #ghdh
        return newx

    def mavg_pool2d(self,x,size):
        b,c,h,w = x.shape
        #newx=(x[:,:,0::2,0::2]+x[:,:,1::2,0::2]+x[:,:,0::2,1::2]+x[:,:,1::2,1::2])/4
        newx=(x[:,:,0::2,0::2])
        return newx


    def forward(self, x, stoch=True):
        if self.training==False:
            stoch=False
        out = F.relu(self.conv1(x))
        if stoch:
            out = self.savg_pool2d(out, 2)
        else:
            out = F.avg_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        if stoch:
            out = self.savg_pool2d(out, 2)
        else:
            out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    
