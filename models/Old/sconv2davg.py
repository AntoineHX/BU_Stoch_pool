import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class SConv2dAvg(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,ceil_mode=True):
        super(SConv2dAvg, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.deconv = nn.ConvTranspose2d(1, 1, kernel_size, 1, padding=0, output_padding=0, groups=1, bias=False, dilation=1, padding_mode='zeros')
        nn.init.constant_(self.deconv.weight, 1)
        self.pooldeconv = nn.ConvTranspose2d(1, 1, kernel_size=stride,padding=0,stride=stride, output_padding=0, groups=1, bias=False, dilation=1, padding_mode='zeros')
        nn.init.constant_(self.pooldeconv.weight, 1)
        self.weight = nn.Parameter(conv.weight)
        self.bias = nn.Parameter(conv.bias)
        self.stride = stride       
        self.dilation = dilation 
        self.padding = padding
        self.kernel_size = kernel_size
        self.ceil_mode = ceil_mode
       
    def forward(self, input, selh=-torch.ones(1,1), selw=-torch.ones(1,1), mask=-torch.ones(1,1),stoch=False,stride=-1):
        device=input.device
        if stride==-1:
            stride = self.stride
        #stoch=True
        if stoch==False:
            stride=1 #test with real average pooling
        batch_size, in_channels, in_h, in_w = input.shape
        out_channels, in_channels, kh, kw =  self.weight.shape

        afterconv_h = in_h-(kh-1) #size after conv
        afterconv_w = in_w-(kw-1)
        if self.ceil_mode:
            out_h = math.ceil(afterconv_h/stride)
            out_w = math.ceil(afterconv_w/stride)
        else:
            out_h = math.floor(afterconv_h/stride)
            out_w = math.floor(afterconv_w/stride)
        unfold = torch.nn.Unfold(kernel_size=(kh, kw), dilation=self.dilation, padding=self.padding, stride=1)
        inp_unf = unfold(input)
        if stride!=1:
            inp_unf = inp_unf.view(batch_size,in_channels*kh*kw,afterconv_h,afterconv_w)
            if selh[0,0]==-1:
                resth = (out_h*stride)-afterconv_h
                restw = (out_w*stride)-afterconv_w
                selh = torch.randint(stride,(out_h,out_w), device=device)
                selw = torch.randint(stride,(out_h,out_w), device=device)
                # print(selh.shape)
                if resth!=0:
                    # Cas : (stride-resth)=0 ?
                    selh[-1,:]=selh[-1,:]%(stride-resth);selh[:,-1]=selh[:,-1]%(stride-restw)
                    selw[-1,:]=selw[-1,:]%(stride-resth);selw[:,-1]=selw[:,-1]%(stride-restw)
            rng_h = selh + torch.arange(0,out_h*stride,stride,device=device).view(-1,1)
            rng_w = selw + torch.arange(0,out_w*stride,stride,device=device)
           
            if mask[0,0]==-1:
                inp_unf = inp_unf[:,:,rng_h,rng_w].view(batch_size,in_channels*kh*kw,-1)
            else:
                inp_unf = inp_unf[:,:,rng_h[mask>0],rng_w[mask>0]]

        #Matrix mul
        if self.bias is None:
            out_unf = inp_unf.transpose(1, 2).matmul(self.weight.view(self.weight.size(0), -1).t()).transpose(1, 2)
        else:
            out_unf = (inp_unf.transpose(1, 2).matmul(self.weight.view(self.weight.size(0), -1).t()) + self.bias).transpose(1, 2)

        if stride==1 or mask[0,0]==-1:
            out = out_unf.view(batch_size,out_channels,out_h,out_w) #Fold
            if stoch==False:
                out = F.avg_pool2d(out,self.stride,ceil_mode=True)
        else:
            out = torch.zeros(batch_size, out_channels,out_h,out_w,device=device)
            out[:,:,mask>0] = out_unf
        return out        

    def comp(self,h,w,mask=-torch.ones(1,1)):
        out_h = (h-(self.kernel_size))/self.stride
        out_w = (w-(self.kernel_size))/self.stride
        if self.ceil_mode:
            out_h = math.ceil(out_h)
            out_w = math.ceil(out_w)
        else:
            out_h = math.floor(out_h)
            out_w = math.florr(out_w)
        if mask[0,0]==-1:
            comp = self.weight.numel()*out_h*out_w 
        else:
            comp = self.weight.numel()*(mask>0).sum()
        return comp

    def sample(self,h,w,mask):
        '''
            h, w : forward input shape
            mask : mask of output used in computation
        '''
        stride = self.stride
        out_channels, in_channels, kh, kw =  self.weight.shape
        device=mask.device

        afterconv_h = h-(kh-1) #Pk afterconv ?
        afterconv_w = w-(kw-1)
        if self.ceil_mode:
            out_h = math.ceil(afterconv_h/stride)
            out_w = math.ceil(afterconv_w/stride)
        else:
            out_h = math.floor(afterconv_h/stride)
            out_w = math.floor(afterconv_w/stride)
        selh = torch.randint(stride,(out_h,out_w), device=device)
        selw = torch.randint(stride,(out_h,out_w), device=device)

        resth = (out_h*stride)-afterconv_h #simplement egale a stride-1, non ?
        restw = (out_w*stride)-afterconv_w
        # print('resth', resth, self.stride)
        if resth!=0:
            selh[-1,:]=selh[-1,:]%(stride-resth);selh[:,-1]=selh[:,-1]%(stride-restw)
            selw[-1,:]=selw[-1,:]%(stride-resth);selw[:,-1]=selw[:,-1]%(stride-restw)
        maskh = (out_h)*stride
        maskw = (out_w)*stride
        rng_h = selh + torch.arange(0,out_h*stride,stride,device=device).view(-1,1)
        rng_w = selw + torch.arange(0,out_w*stride,stride,device=device)
        # rng_w = selw + torch.arange(0,out_w*self.stride,self.stride,device=device).view(-1,1)
        nmask = torch.zeros((maskh,maskw),device=device)
        nmask[rng_h,rng_w] = 1
        #rmask = mask * nmask
        dmask = self.pooldeconv(mask.float().view(1,1,mask.shape[0],mask.shape[1]))
        rmask = nmask * dmask
        #rmask = rmask[:,:,:out_h,:out_w]
        fmask = self.deconv(rmask)
        fmask = fmask[0,0]
        return selh,selw,fmask.long()

    def get_size(self,h,w):
        # newh=(h-(self.kernel_size-1)+(self.stride-1))/self.stride
        # neww=(w-(self.kernel_size-1)+(self.stride-1))/self.stride
        # print(newh,neww)
        newh=math.floor(((h + 2*self.padding - self.dilation*(self.kernel_size-1) - 1)/self.stride) + 1)
        neww=math.floor(((w + 2*self.padding - self.dilation*(self.kernel_size-1) - 1)/self.stride) + 1)
        return newh, neww
