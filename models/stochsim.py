import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# spatial batch and channel
def savg_pool2d_sbc(x,size,ceil_mode=False):
    b,c,h,w = x.shape
    device = x.device
    if ceil_mode:
        out_h = math.ceil(h/size)
        out_w = math.ceil(w/size)
    else:
        out_h = math.floor(h/size)
        out_w = math.floor(w/size)
    selh = torch.randint(size,(b,c,out_h,out_w), device=device)
    #selh[:] = 0
    rngh = torch.arange(0,h,size,device=x.device).view(1,1,-1,1)
    selh = selh+rngh

    selw = torch.randint(size,(b,c,out_h,out_w), device=device)
    #selw[:] = 0
    rngw = torch.arange(0,w,size,device=x.device).view(1,1,1,-1)
    selw = selw+rngw
    selc = torch.arange(0,c,device=x.device).view(1,c,1,1).repeat(b,1,out_h,out_w)
    selb = torch.arange(0,b,device=x.device).view(b,1,1,1).repeat(1,c,out_h,out_w)
    newx = x[selb,selc,selh, selw]
    return newx

#spatial and channel, same for all batch
def savg_pool2d_sc(x,size,ceil_mode=False):
    b,c,h,w = x.shape
    device = x.device
    if ceil_mode:
        out_h = math.ceil(h/size)
        out_w = math.ceil(w/size)
    else:
        out_h = math.floor(h/size)
        out_w = math.floor(w/size)
    selh = torch.randint(size,(c,out_h,out_w), device=device)
    #selh[:] = 0
    rngh = torch.arange(0,h,size,device=x.device).view(1,-1,1)
    selh = selh+rngh

    selw = torch.randint(size,(c,out_h,out_w), device=device)
    #selw[:] = 0
    rngw = torch.arange(0,w,size,device=x.device).view(1,1,-1)
    selw = selw+rngw
    selc = torch.arange(0,c,device=x.device).view(c,1,1).repeat(1,out_h,out_w)

    newx = x[:,selc,selh, selw]
    return newx

#spatial and batch, same for all channels
def savg_pool2d_sb(x,size,ceil_mode=False):
    b,c,h,w = x.shape
    device = x.device
    if ceil_mode:
        out_h = math.ceil(h/size)
        out_w = math.ceil(w/size)
    else:
        out_h = math.floor(h/size)
        out_w = math.floor(w/size)
    selh = torch.randint(size,(b,out_h,out_w), device=device)
    #selh[:] = 0
    rngh = torch.arange(0,h,size,device=x.device).view(1,-1,1)
    selh = selh+rngh

    selw = torch.randint(size,(b,out_h,out_w), device=device)
    #selw[:] = 0
    rngw = torch.arange(0,w,size,device=x.device).view(1,1,-1)
    selw = selw+rngw
    selb = torch.arange(0,b,device=x.device).view(b,1,1).repeat(1,out_h,out_w)

    newx = x.transpose(1,0)
    newx = newx[:,selb,selh, selw]
    return newx.transpose(1,0)

#spatial stochasticity, same for all batch and channels
def savg_pool2d_s(x,size,ceil_mode=False):
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

def savg_pool2d_sdrop(x,size,ceil_mode=False,drop=0,repeat=1):
    b,c,h,w = x.shape
    device = x.device
    if ceil_mode:
        out_h = math.ceil(h/size)
        out_w = math.ceil(w/size)
    else:
        out_h = math.floor(h/size)
        out_w = math.floor(w/size)

    for l in range(repeat):
        selh = torch.randint(size,(out_h,out_w), device=device)
        rngh = torch.arange(0,h,size,device=x.device).view(-1,1)
        selh = selh+rngh

        selw = torch.randint(size,(out_h,out_w), device=device)
        rngw = torch.arange(0,w,size,device=x.device)
        selw = selw+rngw

        if l==0:
            newx = x[:,:, selh, selw]
        else:
            newx = newx + x[:,:, selh, selw]
    newx = newx/repeat
    if drop!=0:
        dropmask = torch.rand((c), device=device)
        newx[:,dropmask<drop] = 0
    return newx


def savg_pool2d(x, stride,mode,ceil_mode=False,repeat=1):
    if mode=='s':
        out = savg_pool2d_s(x,stride,ceil_mode=ceil_mode)
    if mode=='sdrop':
        out = savg_pool2d_sdrop(x,stride,ceil_mode=ceil_mode,repeat=repeat)
    elif mode =='sb':
        out = savg_pool2d_sb(x,stride,ceil_mode=ceil_mode)
    elif mode =='sc':
        out = savg_pool2d_sc(x,stride,ceil_mode=ceil_mode)
    elif mode =='sbc':
        out = savg_pool2d_sbc(x,stride,ceil_mode=ceil_mode)
    return out



