
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import opt_einsum as oe

class SConv2dStride(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,ceil_mode=True,bias=False):    
        super(SConv2dStride, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size , stride=stride, padding=padding,dilation=dilation,bias=bias)
        self.stride = stride
        self.ceil_mode = ceil_mode

    def forward(self, x,stoch = True):
        stoch=True #for some reason average does not work...
        if stoch:
            device= x.device
            selh = torch.randint(self.conv.stride[0],(1,), device=device)[0]
            selw = torch.randint(self.conv.stride[1],(1,), device=device)[0]
            out = self.conv(x[:,:,selh:,selw:])       
        else:
            self.conv.stride = (1,1)
            out = self.conv(x)        
            out = F.avg_pool2d(out,self.stride,ceil_mode=self.ceil_mode)
            self.conv.stride = (self.stride,self.stride)
        return out

class SConv2dAvg(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,ceil_mode=True, bias = True):
        super(SConv2dAvg, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.deconv = nn.ConvTranspose2d(1, 1, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False, dilation=1, padding_mode='zeros')
        nn.init.constant_(self.deconv.weight, 1)
        self.pooldeconv = nn.ConvTranspose2d(1, 1, kernel_size=stride,padding=0,stride=stride, output_padding=0, groups=1, bias=False, dilation=1, padding_mode='zeros')
        nn.init.constant_(self.pooldeconv.weight, 1)
        self.weight = nn.Parameter(conv.weight)
        if bias:
            self.bias = nn.Parameter(conv.bias)
        else:
            self.bias = None
        self.stride = stride       
        self.dilation = dilation 
        self.padding = padding
        self.kernel_size = kernel_size
        self.ceil_mode = ceil_mode

    
    def forward_fast(self, input, index=-torch.ones(1), mask=-torch.ones(1,1),stoch=True,stride=-1): #ceil_mode = True not right
        device=input.device
        if stride==-1:
            stride = self.stride #if stride not defined use self.stride
        if stoch==False:
            stride=1 #test with real average pooling
        batch_size, in_channels, in_h, in_w = input.shape
        out_channels, in_channels, kh, kw =  self.weight.shape
        afterconv_h,afterconv_w,out_h,out_w = self.get_size(in_h,in_w,stride)
        
        unfold = torch.nn.Unfold(kernel_size=(kh, kw), dilation=self.dilation, padding=self.padding, stride=1)
        inp_unf = unfold(input) #transform into a matrix (batch_size, in_channels*kh*kw,afterconv_h,afterconv_w)

        if index[0]==-1 and stride!=1: #or stride!=1:    
            index,mask = self.sample(in_h,in_w,batch_size,device,mask)

            if mask[0,0]==-1:# in case of not given mask use only sampled selection
                #inp_unf = inp_unf[:,:,rng_h,rng_w].view(batch_size,in_channels*kh*kw,-1)
                inp_unf = torch.gather(inp_unf.view(batch_size,in_channels*kh*kw,afterconv_h*afterconv_w),2,index.view(batch_size,in_channels*kh*kw,out_h*out_w)).view(batch_size,in_channels*kh*kw,out_h*out_w)
            else:#in case of a valid mask use selection only on the mask locations
                inp_unf = inp_unf[:,:,rng_h[mask>0],rng_w[mask>0]]

        #Matrix mul
        if self.bias is None:
            #flt = self.weight.view(self.weight.size(0), -1).t()
            #out_unf = inp_unf.transpose(2,1).matmul(flt).transpose(1, 2)
            out_unf = oe.contract('bji,kj->bki',inp_unf,self.weight.view(self.weight.size(0), -1),backend='torch')
            #print(((out_unf-out_unf1)**2).mean())
        else:
            #out_unf = oe.contract('bji,kj,k->bki',inp_unf,self.weight.view(self.weight.size(0), -1),self.bias,backend='torch')#+self.bias.view(1,-1,1)#wrong
            out_unf = oe.contract('bji,kj->bki',inp_unf,self.weight.view(self.weight.size(0), -1),backend='torch')+self.bias.view(1,-1,1)#sligthly slower but correct
            #out_unf1 = (inp_unf.transpose(1, 2).matmul(self.weight.view(self.weight.size(0), -1).t()) + self.bias).transpose(1, 2)
            #print(((out_unf-out_unf1)**2).mean())
            #self.flt = self.weight.view(self.weight.size(0), -1).t() 
            #out_unf = (inp_unf.transpose(1, 2).matmul(self.flt) + self.bias).transpose(1, 2)

        if stride==1 or mask[0,0]==-1:# in case of no mask and stride==1 
            out = out_unf.view(batch_size,out_channels,out_h,out_w) #Fold
            if stoch==False: #this is done outside for more clarity
                out = F.avg_pool2d(out,self.stride,ceil_mode=True)
        else:#in case of mask
            out = torch.zeros(batch_size, out_channels,out_h,out_w,device=device)
            out[:,:,mask>0] = out_unf
        return out        

       
    def forward_test(self, input, selh=-torch.ones(1,1), selw=-torch.ones(1,1), mask=-torch.ones(1,1),stoch=True,stride=-1):#ugly but faster
        device=input.device
        if stride==-1:
            stride = self.stride #if stride not defined use self.stride
        if stoch==False:
            stride=1 #test with real average pooling
        batch_size, in_channels, in_h, in_w = input.shape
        out_channels, in_channels, kh, kw =  self.weight.shape
    
        #afterconv_h,afterconv_w,out_h,out_w = self.get_size(in_h,in_w)
        #if selh[0,0]==-1:    
        #    index,mask = self.sample(in_h,in_w,batch_size,device,mask)

        if 1:
            afterconv_h = in_h+2*self.padding-(kh-1) #size after conv
            afterconv_w = in_w+2*self.padding-(kw-1)
            if self.ceil_mode: #ceil_mode = talse default mode for strided conv
                out_h = math.ceil(afterconv_h/stride)
                out_w = math.ceil(afterconv_w/stride)
            else: #ceil_mode = false default mode for pooling
                out_h = math.floor(afterconv_h/stride)
                out_w = math.floor(afterconv_w/stride)
        unfold = torch.nn.Unfold(kernel_size=(kh, kw), dilation=self.dilation, padding=self.padding, stride=1)
        inp_unf = unfold(input) #transform into a matrix (batch_size, in_channels*kh*kw,afterconv_h,afterconv_w)
        if 1:
            if stride!=1: # if stride==1 there is no pooling
                inp_unf = inp_unf.view(batch_size,in_channels*kh*kw,afterconv_h,afterconv_w)
                if selh[0,0]==-1: # if not given sampled selection
                    #selction of where to sample for each pooling location
                    sel = torch.randint(stride*stride,(out_h,out_w), device=device)

                    if self.ceil_mode: #in case of ceil_mode need to select only the good locations for the last regions
                        resth = (out_h*stride)-afterconv_h
                        restw = (out_w*stride)-afterconv_w                
                        if resth!=0:
                            sel[-1] = (sel[-1]//stride)%(stride-resth)*stride+(sel[-1]%stride)
                            sel[:,-1] = (sel[:,-1]%stride)%(stride-restw)+sel[:,-1]//stride*stride
                            #print(stride-resth,sel[-1])
                            #print(stride-restw,sel[:,-1])

                #rng = torch.arange(0,afterconv_h*afterconv_w,stride*stride,device=device).view(out_h,out_w)
                rng = torch.arange(0,out_h*stride*out_w*stride,stride*stride,device=device).view(out_h,out_w)
                index = sel+rng
                index = index.repeat(batch_size,in_channels*kh*kw,1,1)           


                if mask[0,0]==-1:# in case of not given mask use only sampled selection
                    #inp_unf = inp_unf[:,:,rng_h,rng_w].view(batch_size,in_channels*kh*kw,-1)
                    inp_unf = torch.gather(inp_unf.view(batch_size,in_channels*kh*kw,afterconv_h*afterconv_w),2,index.view(batch_size,in_channels*kh*kw,out_h*out_w)).view(batch_size,in_channels*kh*kw,out_h*out_w)
                else:#in case of a valid mask use selection only on the mask locations
                    inp_unf = inp_unf[:,:,rng_h[mask>0],rng_w[mask>0]]

        #Matrix mul
        if self.bias is None:
            #flt = self.weight.view(self.weight.size(0), -1).t()
            #out_unf = inp_unf.transpose(2,1).matmul(flt).transpose(1, 2)
            out_unf = oe.contract('bji,kj->bki',inp_unf,self.weight.view(self.weight.size(0), -1),backend='torch')
            #print(((out_unf-out_unf1)**2).mean())
        else:
            #out_unf = oe.contract('bji,kj,b->bki',inp_unf,self.weight.view(self.weight.size(0), -1),self.bias,backend='torch')#+self.bias.view(1,-1,1)#still slow
            out_unf = oe.contract('bji,kj->bki',inp_unf,self.weight.view(self.weight.size(0), -1),backend='torch')+self.bias.view(1,-1,1)#still slow
            #self.flt = self.weight.view(self.weight.size(0), -1).t() 
            #out_unf = (inp_unf.transpose(1, 2).matmul(self.flt) + self.bias).transpose(1, 2)

        if stride==1 or mask[0,0]==-1:# in case of no mask and stride==1 
            out = out_unf.view(batch_size,out_channels,out_h,out_w) #Fold
            if stoch==False: #this is done outside for more clarity
                out = F.avg_pool2d(out,self.stride,ceil_mode=True)
        else:#in case of mask
            out = torch.zeros(batch_size, out_channels,out_h,out_w,device=device)
            out[:,:,mask>0] = out_unf
        return out        


    def forward(self, input, selh=-torch.ones(1,1), selw=-torch.ones(1,1), mask=-torch.ones(1,1),stoch=True,stride=-1):
        device=input.device
        if stride==-1:
            stride = self.stride #if stride not defined use self.stride
        if stoch==False:
            stride=1 #test with real average pooling
        batch_size, in_channels, in_h, in_w = input.shape
        out_channels, in_channels, kh, kw =  self.weight.shape
    
        afterconv_h = in_h+2*self.padding-(kh-1) #size after conv
        afterconv_w = in_w+2*self.padding-(kw-1)
        if self.ceil_mode: #ceil_mode = talse default mode for strided conv
            out_h = math.ceil(afterconv_h/stride)
            out_w = math.ceil(afterconv_w/stride)
        else: #ceil_mode = false default mode for pooling
            out_h = math.floor(afterconv_h/stride)
            out_w = math.floor(afterconv_w/stride)
        unfold = torch.nn.Unfold(kernel_size=(kh, kw), dilation=self.dilation, padding=self.padding, stride=1)
        inp_unf = unfold(input) #transform into a matrix (batch_size, in_channels*kh*kw,afterconv_h,afterconv_w)
        if stride!=1: # if stride==1 there is no pooling
            inp_unf = inp_unf.view(batch_size,in_channels*kh*kw,afterconv_h,afterconv_w)
            if selh[0,0]==-1: # if not given sampled selection
                #selction of where to sample for each pooling location
                selh = torch.randint(stride,(out_h,out_w), device=device)
                selw = torch.randint(stride,(out_h,out_w), device=device)
                
                resth = (out_h*stride)-afterconv_h
                restw = (out_w*stride)-afterconv_w                
                if resth!=0 and self.ceil_mode: #in case of ceil_mode need to select only the good locations for the last regions
                    selh[-1,:]=selh[-1,:]%(stride-resth);selh[:,-1]=selh[:,-1]%(stride-restw)
                    selw[-1,:]=selw[-1,:]%(stride-resth);selw[:,-1]=selw[:,-1]%(stride-restw)
            #the postion should be global by adding range...
            rng_h = selh + torch.arange(0,out_h*stride,stride,device=device).view(-1,1)
            rng_w = selw + torch.arange(0,out_w*stride,stride,device=device)
           
            if mask[0,0]==-1:# in case of not given mask use only sampled selection
                inp_unf = inp_unf[:,:,rng_h,rng_w].view(batch_size,in_channels*kh*kw,-1)
            else:#in case of a valid mask use selection only on the mask locations
                inp_unf = inp_unf[:,:,rng_h[mask>0],rng_w[mask>0]]

        #Matrix mul
        if self.bias is None:
            #out_unf = inp_unf.transpose(1, 2).matmul(self.weight.view(self.weight.size(0), -1).t()).transpose(1, 2)
            out_unf = oe.contract('bji,kj->bki',inp_unf,self.weight.view(self.weight.size(0), -1),backend='torch')
        else:
            out_unf = oe.contract('bji,kj->bki',inp_unf,self.weight.view(self.weight.size(0), -1),backend='torch')+self.bias.view(1,-1,1)
            #out_unf = (inp_unf.transpose(1, 2).matmul(self.weight.view(self.weight.size(0), -1).t()) + self.bias).transpose(1, 2)

        if stride==1 or mask[0,0]==-1:# in case of no mask and stride==1 
            out = out_unf.view(batch_size,out_channels,out_h,out_w) #Fold
            if stoch==False: #this is done outside for more clarity
                out = F.avg_pool2d(out,self.stride,ceil_mode=True)
        else:#in case of mask
            out = torch.zeros(batch_size, out_channels,out_h,out_w,device=device)
            out[:,:,mask>0] = out_unf
        return out        

    def forward_slowwithbatch(self, input, selh=-torch.ones(1,1), selw=-torch.ones(1,1), mask=-torch.ones(1,1),stoch=True,stride=-1):
        device=input.device
        if stride==-1:
            stride = self.stride
        #stoch=True
        if stoch==False:
            stride=1 #test with real average pooling
        batch_size, in_channels, in_h, in_w = input.shape
        out_channels, in_channels, kh, kw =  self.weight.shape

        afterconv_h = in_h+2*padding-(kh-1) #size after conv
        afterconv_w = in_w+2*padding-(kw-1)
        if self.ceil_mode:
            out_h = math.ceil(afterconv_h/stride)
            out_w = math.ceil(afterconv_w/stride)
        else:
            out_h = math.floor(afterconv_h/stride)
            out_w = math.floor(afterconv_w/stride)
        unfold = torch.nn.Unfold(kernel_size=(kh, kw), dilation=self.dilation, padding=self.padding, stride=1)
        inp_unf = unfold(input)
        if stride!=1:
            inp_unf = inp_unf.view(batch_size,in_channels,kh*kw,afterconv_h,afterconv_w)
            if selh[0,0]==-1:
                resth = (out_h*stride)-afterconv_h
                restw = (out_w*stride)-afterconv_w
                selh = torch.randint(stride,(in_channels,out_h,out_w), device=device)
                selw = torch.randint(stride,(in_channels,out_h,out_w), device=device)
                # print(selh.shape)
                if resth!=0:
                    # Cas : (stride-resth)=0 ?
                    selh[-1,:]=selh[-1,:]%(stride-resth);selh[:,-1]=selh[:,-1]%(stride-restw)
                    selw[-1,:]=selw[-1,:]%(stride-resth);selw[:,-1]=selw[:,-1]%(stride-restw)
            rng_h = selh + torch.arange(0,out_h*stride,stride,device=device).view(1,-1,1)
            rng_w = selw + torch.arange(0,out_w*stride,stride,device=device).view(1,1,-1)
            selc = torch.arange(0,in_channels,device=input.device).view(in_channels,1,1).repeat(1,out_h,out_w)           

            if mask[0,0]==-1:
                inp_unf = inp_unf.transpose(1,2)[:,:,selc,rng_h,rng_w].transpose(2,1).reshape(batch_size,in_channels*kh*kw,-1)
            else:
                inp_unf = inp_unf[:,:,rng_h[mask>0],rng_w[mask>0]]

        #Matrix mul
        if self.bias is None:
            out_unf = inp_unf.transpose(1, 2).matmul(self.weight.view(self.weight.size(0), -1).t()).transpose(1, 2)
        else:
            out_unf = (inp_unf.transpose(1, 2).matmul(self.weight.view(self.weight.size(0), -1).t()) + self.bias).transpose(1, 2)

        if stride==1 or mask[0,0]==-1:
            out = out_unf.view(batch_size,out_channels,out_h,out_w) #Fold
        #    if stoch==False:
        #        out = F.avg_pool2d(out,self.stride,ceil_mode=True)
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

    def sample_slow(self,h,w,mask):
        '''
            h, w : forward input shape
            mask : mask of output used in computation
        '''
        stride = self.stride
        out_channels, in_channels, kh, kw =  self.weight.shape
        device=mask.device

        #Shape after simple forward conv ?
        afterconv_h = h+2*padding-(kh-1) 
        afterconv_w = w+2*padding-(kw-1)
        # print(afterconv_h)
        # print(afterconv_h/stride)

        #Shape after forward ? (== mask.shape ?) #Padding, Dilatation pas pris en compte ?
        if self.ceil_mode:
            out_h = math.ceil(afterconv_h/stride)
            out_w = math.ceil(afterconv_w/stride)
        else:
            out_h = math.floor(afterconv_h/stride)
            out_w = math.floor(afterconv_w/stride)

        #selh = torch.randint(stride,(out_h,out_w), device=device)
        #selw = torch.randint(stride,(out_h,out_w), device=device)

        resth = (out_h*stride)-afterconv_h #reste de ceil/floor, 0 ou 1
        restw = (out_w*stride)-afterconv_w
        # print('rest', resth, restw)
        if resth!=0:
            selh[-1,:]=selh[-1,:]%(stride-resth);selh[:,-1]=selh[:,-1]%(stride-restw)
            selw[-1,:]=selw[-1,:]%(stride-resth);selw[:,-1]=selw[:,-1]%(stride-restw)
        maskh = (out_h)*stride
        maskw = (out_w)*stride
        # print('mask', maskh, maskw)
        rng_h = selh + torch.arange(0,out_h*stride,stride,device=device).view(-1,1)
        rng_w = selw + torch.arange(0,out_w*stride,stride,device=device)
        # rng_w = selw + torch.arange(0,out_w*self.stride,self.stride,device=device).view(-1,1)
        nmask = torch.zeros((maskh,maskw),device=device)
        nmask[rng_h,rng_w] = 1
        #rmask = mask * nmask
        dmask = self.pooldeconv(mask.float().view(1,1,mask.shape[0],mask.shape[1]))
        rmask = nmask * dmask
        #rmask = rmask[:,:,:out_h,:out_w]
        # print('rmask', rmask.shape)
        fmask = self.deconv(rmask)
        # print('fmask', fmask.shape)
        fmask = fmask[0,0]
        return selh,selw,fmask.long()

    def sample(self,in_h,in_w,batch_size,device,mask=-torch.ones(1,1)):
        '''
            h, w : forward input shape
            mask : mask of output used in computation
        '''
        stride = self.stride
        out_channels, in_channels, kh, kw =  self.weight.shape
        #device=mask.device

        #Shape after simple forward conv ?
        afterconv_h = in_h+2*self.padding-(kh-1) #size after conv
        afterconv_w = in_w+2*self.padding-(kw-1)

        #Shape after forward ? (== mask.shape ?) #Padding, Dilatation pas pris en compte ?
        if self.ceil_mode:
            out_h = math.ceil(afterconv_h/stride)
            out_w = math.ceil(afterconv_w/stride)
        else:
            out_h = math.floor(afterconv_h/stride)
            out_w = math.floor(afterconv_w/stride)

        sel = torch.randint(stride*stride,(out_h,out_w), device=device)

        if self.ceil_mode: #in case of ceil_mode need to select only the good locations for the last regions
            resth = (out_h*stride)-afterconv_h
            restw = (out_w*stride)-afterconv_w                
            if resth!=0:
                sel[-1] = (sel[-1]//stride)%(stride-resth)*stride+(sel[-1]%stride)
                sel[:,-1] = (sel[:,-1]%stride)%(stride-restw)+sel[:,-1]//stride*stride

        rng = torch.arange(0,out_h*stride*out_w*stride,stride*stride,device=device).view(out_h,out_w)
        index = sel+rng
        index = index.repeat(batch_size,in_channels*kh*kw,1,1)           

        #inp_unf = torch.gather(inp_unf.view(batch_size,in_channels*kh*kw,afterconv_h*afterconv_w),2,index.view(batch_size,in_channels*kh*kw,out_h*out_w)).view(batch_size,in_channels*kh*kw,out_h*out_w)
        
        if mask[0,0]!=-1:
            maskh = (out_h)*stride
            maskw = (out_w)*stride        
            nmask = torch.zeros((maskh,maskw),device=device)
            nmask[rng_h,rng_w] = 1
            #rmask = mask * nmask
            dmask = self.pooldeconv(mask.float().view(1,1,mask.shape[0],mask.shape[1]))
            rmask = nmask * dmask
            #rmask = rmask[:,:,:out_h,:out_w]
            # print('rmask', rmask.shape)
            fmask = self.deconv(rmask)
            # print('fmask', fmask.shape)
            mask = fmask[0,0].long()
            
        return index,mask#.long()


    def get_size(self,in_h,in_w,stride=-1):
        
        if stride==-1:
            stride = self.stride
        out_channels, in_channels, kh, kw =  self.weight.shape
        afterconv_h = in_h+2*self.padding-(kh-1) #size after conv
        afterconv_w = in_w+2*self.padding-(kw-1)
        if self.ceil_mode: #ceil_mode = talse default mode for strided conv
            out_h = math.ceil(afterconv_h/stride)
            out_w = math.ceil(afterconv_w/stride)
        else: #ceil_mode = false default mode for pooling
            out_h = math.floor(afterconv_h/stride)
            out_w = math.floor(afterconv_w/stride)
        #newh=math.floor(((h + 2*self.padding - self.dilation*(self.kernel_size-1) - 1)/self.stride) + 1)
        #neww=math.floor(((w + 2*self.padding - self.dilation*(self.kernel_size-1) - 1)/self.stride) + 1)
        return afterconv_h,afterconv_w,out_h,out_w

