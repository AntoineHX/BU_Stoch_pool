'''LeNet in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyLeNet2(nn.Module):
    def __init__(self):
        super(MyLeNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 60, 5)
        self.conv2 = nn.Conv2d(60, 160, 5)
        self.fc1   = nn.Linear(160*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)        

# Vanilla Convolution
    def myconv2d(self, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        batch_size, in_channels, in_h, in_w = input.shape
        out_channels, in_channels, kh, kw =  weight.shape
        out_h = in_h-2*(int(kh)/2)
        out_w = in_w-2*(int(kw)/2)

        unfold = torch.nn.Unfold(kernel_size=(kh, kw), dilation=dilation, padding=padding, stride=stride)
        inp_unf = unfold(input)#.view(batch_size,in_channels*kh*kw,out_h,out_w)
                

        if bias is None:
            out_unf = inp_unf.transpose(1, 2).matmul(weight.view(weight.size(0), -1).t()).transpose(1, 2)
        else:
            out_unf = (inp_unf.transpose(1, 2).matmul(weight.view(weight.size(0), -1).t()) + bias).transpose(1, 2)

        out = out_unf.view(batch_size, out_channels, out_h, out_w)
        return out

    def myconv2d_avg(self, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1,size=2):
        batch_size, in_channels, in_h, in_w = input.shape
        out_channels, in_channels, kh, kw =  weight.shape
        out_h = in_h-2*(int(kh)/2)
        out_w = in_w-2*(int(kw)/2)

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
        #out = F.relu(self.conv1(x))
        out = F.relu(self.myconv2d(x, self.conv1.weight, bias=self.conv1.bias))
        if stoch:
            out = self.savg_pool2d(out, 2)
        else:
            out = F.avg_pool2d(out, 2)
        #out = F.relu(self.conv2(out))
        if 0:
            out = F.relu(self.myconv2d_avg(out, self.conv2.weight, bias=self.conv2.bias,size=2)) 
        else:
            #out = F.relu(self.conv2(out))
            out = F.relu(self.myconv2d(out, self.conv2.weight, bias=self.conv2.bias))     
            out = F.avg_pool2d(out, 2)
        #if stoch:
        #    out = self.savg_pool2d(out, 2)
        #else:
        #    out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1 )
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    
