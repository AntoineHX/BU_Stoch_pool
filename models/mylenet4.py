'''LeNet in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from .stoch import SConv2dAvg
from .stochsim import savg_pool2d

class MyLeNetNormal(nn.Module):#epoch 12s
    def __init__(self):
        super(MyLeNetNormal, self).__init__()
        self.conv1 = nn.Conv2d(3, 200, 3, stride=1)
        self.conv2 = nn.Conv2d(200, 400, 3, stride=1)
        self.conv3 = nn.Conv2d(400, 800, 3, stride=1)
        self.conv4 = nn.Conv2d(800, 10, 3, stride=1)
        #self.fc1   = nn.Linear(800, 10)

    def forward(self, x, stoch=True):

        out = F.relu(self.conv1(x))
        out = F.avg_pool2d(out,2,ceil_mode=True)
        out = F.relu(self.conv2(out))
        out = F.avg_pool2d(out,2,ceil_mode=True)
        out = F.relu(self.conv3(out))
        out = F.avg_pool2d(out,2,ceil_mode=True)
        out = self.conv4(out)
        #out = F.avg_pool2d(out,2,ceil_mode=True)

        out = out.view(out.size(0), -1 )
        #out = (self.fc1(out))
        return out

class MyLeNetSimNormal(nn.Module):#epoch 12s
    def __init__(self):
        super(MyLeNetSimNormal, self).__init__()
        self.conv1 = nn.Conv2d(3, 200, 3, stride=1)
        self.conv2 = nn.Conv2d(200, 400, 3, stride=1)
        self.conv3 = nn.Conv2d(400, 800, 3, stride=1)
        self.conv4 = nn.Conv2d(800, 10, 3, stride=1)
        #self.fc1   = nn.Linear(800, 10)

    def forward(self, x, stoch=True):

        out = F.relu(self.conv1(x))
        # out = self.savg_pool2d(out,2,ceil_mode=True)
        out = savg_pool2d(out,2, mode='s', ceil_mode=True)
        out = F.relu(self.conv2(out))
        # out = self.savg_pool2d(out,2,ceil_mode=True)
        out = savg_pool2d(out,2, mode='s', ceil_mode=True)
        out = F.relu(self.conv3(out))
        # out = self.savg_pool2d(out,2,ceil_mode=True)
        out = savg_pool2d(out,2, mode='s', ceil_mode=True)
        out = self.conv4(out)
        #out = F.avg_pool2d(out,2,ceil_mode=True)

        out = out.view(out.size(0), -1 )
        #out = (self.fc1(out))
        return out


class MyLeNetStride(nn.Module):#epoch 6s
    def __init__(self):
        super(MyLeNetStride, self).__init__()
        self.conv1 = nn.Conv2d(3, 200, 3, stride=2)
        self.conv2 = nn.Conv2d(200, 400, 3, stride=2)
        self.conv3 = nn.Conv2d(400, 800, 3, stride=2)
        self.conv4 = nn.Conv2d(800, 10, 3, stride=1)
        #self.fc1   = nn.Linear(800, 10)

    def forward(self, x, stoch=True):

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = self.conv4(out)

        out = out.view(out.size(0), -1 )
        #out = (self.fc1(out))
        return out

class MyLeNetMatNormal(nn.Module):#epach 21s
    def __init__(self):
        super(MyLeNetMatNormal, self).__init__()
        self.conv1 = SConv2dAvg(3, 200, 3, stride=1)
        self.conv2 = SConv2dAvg(200, 400, 3, stride=1)
        self.conv3 = SConv2dAvg(400, 800, 3, stride=1)
        self.conv4 = SConv2dAvg(800, 10, 3, stride=1)
        #self.fc1   = nn.Linear(800, 10)

    def forward(self, x, stoch=True):
        out = F.relu(self.conv1(x))
        out = F.avg_pool2d(out,2,ceil_mode=True)
        out = F.relu(self.conv2(out))
        out = F.avg_pool2d(out,2,ceil_mode=True)
        out = F.relu(self.conv3(out))
        out = F.avg_pool2d(out,2,ceil_mode=True)
        out = (self.conv4(out))
        #out = F.avg_pool2d(out,1,ceil_mode=True)

        out = out.view(out.size(0), -1 )
        #out = (self.fc1(out))
        return out

class MyLeNetMatStoch(nn.Module):#epoch 17s
    def __init__(self):
        super(MyLeNetMatStoch, self).__init__()
        self.conv1 = SConv2dAvg(3, 200, 3, stride=2)
        self.conv2 = SConv2dAvg(200, 400, 3, stride=2)
        self.conv3 = SConv2dAvg(400, 800, 3, stride=2)
        self.conv4 = SConv2dAvg(800, 10, 3, stride=1)
        #self.fc1   = nn.Linear(800, 10)

    def forward(self, x, stoch=True):
        #print('in',x.shape)
        out = F.relu(self.conv1(x,stoch=stoch))
        #print('c1',out.shape)
        out = F.relu(self.conv2(out,stoch=stoch))
        #print('c2', out.shape)
        out = F.relu(self.conv3(out,stoch=stoch))
        #print('c3',out.shape)
        out = self.conv4(out,stoch=stoch)
        #print('c4',out.shape)
        out = out.view(out.size(0), -1 )
        #out = self.fc1(out)
        return out

class MyLeNetMatNormalNoceil(nn.Module):#epoch 136s 16GB
    def __init__(self,k=3):
        super(MyLeNetMatNormalNoceil, self).__init__()
        self.conv1 = SConv2dAvg(3, 200*k, 3, stride=1,padding=1,ceil_mode=False)
        self.conv2 = SConv2dAvg(200*k, 400*k, 3, stride=1,padding=1,ceil_mode=False)
        self.conv3 = SConv2dAvg(400*k, 800*k, 3, stride=1,padding=1,ceil_mode=False)
        self.conv4 = SConv2dAvg(800*k, 1600*k, 3, stride=1,padding=1,ceil_mode=False)
        self.fc1   = nn.Linear(1600*k, 10)

    def forward(self, x, stoch=True):
        out = F.relu(self.conv1(x,stoch=stoch))
        out = F.avg_pool2d(out,2,ceil_mode=False)
        out = F.relu(self.conv2(out,stoch=stoch))
        out = F.avg_pool2d(out,2,ceil_mode=False)
        out = F.relu(self.conv3(out,stoch=stoch))
        out = F.avg_pool2d(out,2,ceil_mode=False)
        out = F.relu(self.conv4(out,stoch=stoch))
        out = F.avg_pool2d(out,4,ceil_mode=False)
        out = out.view(out.size(0), -1 )
        out = self.fc1(out)
        return out

class MyLeNetMatStochNoceil(nn.Module):#epoch 41s 16BG
    def __init__(self,k=3):
        super(MyLeNetMatStochNoceil, self).__init__()
        self.conv1 = SConv2dAvg(3, 200*k, 3, stride=2,padding=1,ceil_mode=False)
        self.conv2 = SConv2dAvg(200*k, 400*k, 3, stride=2,padding=1,ceil_mode=False)
        self.conv3 = SConv2dAvg(400*k, 800*k, 3, stride=2,padding=1,ceil_mode=False)
        self.conv4 = SConv2dAvg(800*k, 1600*k, 3, stride=4,padding=1,ceil_mode=False)
        self.fc1   = nn.Linear(1600*k, 10)

    def forward(self, x, stoch=True):
        if stoch:
            out = F.relu(self.conv1(x,stoch=stoch))
            out = F.relu(self.conv2(out,stoch=stoch))
            out = F.relu(self.conv3(out,stoch=stoch))
            out = F.relu(self.conv4(out,stoch=stoch))
            out = out.view(out.size(0), -1 )
            out = self.fc1(out)
        else:
            out = F.relu(self.conv1(x,stride=1))
            out = F.avg_pool2d(out,2,ceil_mode=False)
            out = F.relu(self.conv2(out,stride=1))
            out = F.avg_pool2d(out,2,ceil_mode=False)
            out = F.relu(self.conv3(out,stride=1))
            out = F.avg_pool2d(out,2,ceil_mode=False)
            out = F.relu(self.conv4(out,stride=1))
            out = F.avg_pool2d(out,4,ceil_mode=False)
            out = out.view(out.size(0), -1 )
            out = self.fc1(out)

        return out
    
class MyLeNetMatStochBUNoceil(nn.Module):#30.5s 14GB
    def __init__(self,k=3):
        super(MyLeNetMatStochBUNoceil, self).__init__()
        self.conv1 = SConv2dAvg(3, 200*k, 3, stride=2,padding=1,ceil_mode=False)
        self.conv2 = SConv2dAvg(200*k, 400*k, 3, stride=2,padding=1,ceil_mode=False)
        self.conv3 = SConv2dAvg(400*k, 800*k, 3, stride=2,padding=1,ceil_mode=False)
        self.conv4 = SConv2dAvg(800*k, 1600*k, 3, stride=4,padding=1,ceil_mode=False)
        self.fc1   = nn.Linear(1600*k, 10)

    def forward(self, x, stoch=True):
        if stoch:
            #get sizes
            batch_size = x.shape[0]
            device = x.device
            h0,w0 = x.shape[2],x.shape[3]
            _,_,h1,w1 = self.conv1.get_size(h0,w0)
            _,_,h2,w2 = self.conv2.get_size(h1,w1)
            _,_,h3,w3 = self.conv3.get_size(h2,w2)
            _,_,h4,w4 = self.conv4.get_size(h3,w3)
            # print(h0,w0)
            # print(h1,w1)
            # print(h2,w2)
            # print(h3,w3)

            #sample BU
            mask4 = torch.ones(h4,w4).to(x.device)
            # print(mask3.shape)
            index4,mask3 = self.conv4.sample(h3,w3,batch_size,device,mask4)
            index3,mask2 = self.conv3.sample(h2,w2,batch_size,device,mask3)
            index2,mask1 = self.conv2.sample(h1,w1,batch_size,device,mask2)
            index1,mask0 = self.conv1.sample(h0,w0,batch_size,device,mask1)        
            
            ##forward
            out = F.relu(self.conv1(x,index1,mask1,stoch=stoch))
            out = F.relu(self.conv2(out,index2,mask2,stoch=stoch))
            out = F.relu(self.conv3(out,index3,mask3,stoch=stoch))
            out = F.relu(self.conv4(out,index4,mask4,stoch=stoch))
            out = out.view(out.size(0), -1 )
            out = self.fc1(out)
        else:
            out = F.relu(self.conv1(x,stride=1))
            out = F.avg_pool2d(out,2,ceil_mode=False)
            out = F.relu(self.conv2(out,stride=1))
            out = F.avg_pool2d(out,2,ceil_mode=False)
            out = F.relu(self.conv3(out,stride=1))
            out = F.avg_pool2d(out,2,ceil_mode=False)
            out = F.relu(self.conv4(out,stride=1))
            out = F.avg_pool2d(out,4,ceil_mode=False)
            out = out.view(out.size(0), -1 )
            out = self.fc1(out)
        return out

class MyLeNetMatStochBU(nn.Module):#epoch 11s 
    def __init__(self):
        super(MyLeNetMatStochBU, self).__init__()
        self.conv1 = SConv2dAvg(3, 200*k, 3, stride=2)
        self.conv2 = SConv2dAvg(200, 400, 3, stride=2)
        self.conv3 = SConv2dAvg(400, 800, 3, stride=2, ceil_mode=True)
        self.conv4 = SConv2dAvg(800, 10, 3, stride=1)
        # self.fc1   = nn.Linear(800, 10)

    def forward(self, x, stoch=True):
        #get sizes
        h0,w0 = x.shape[2],x.shape[3]
        h1,w1 = self.conv1.get_size(h0,w0)
        h2,w2 = self.conv2.get_size(h1,w1)
        h3,w3 = self.conv3.get_size(h2,w2)
        # print(h0,w0)
        # print(h1,w1)
        # print(h2,w2)
        # print(h3,w3)

        #sample BU
        mask3 = torch.ones(h3,w3).to(x.device)
        # print(mask3.shape)
        selh3,selw3,mask2 = self.conv3.sample(h2,w2,mask=mask3) #Mask2.shape != (h2,w2) ???
        # print(mask2.shape)
        selh2,selw2,mask1 = self.conv2.sample(h1,w1,mask=mask2)
        # print(mask1.shape)
        selh1,selw1,mask0 = self.conv1.sample(h0,w0,mask=mask1)
        #forward
        out = F.relu(self.conv1(x,selh1,selw1,mask1,stoch=stoch))
        out = F.relu(self.conv2(out,selh2,selw2,mask2,stoch=stoch))
        out = F.relu(self.conv3(out,selh3,selw3,mask3,stoch=stoch))

        out = self.conv4(out,stoch=stoch)
        out = out.view(out.size(0), -1 )
        # out = (self.fc1(out))
        return out

# class SConv2dAvg(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
#         super(SConv2dAvg, self).__init__()
#         conv = nn.Conv2d(in_channels, out_channels, kernel_size)
#         self.deconv = nn.ConvTranspose2d(1, 1, kernel_size, 1, padding=0, output_padding=0, groups=1, bias=False, dilation=1, padding_mode='zeros')
#         nn.init.constant_(self.deconv.weight, 1)
#         self.pooldeconv = nn.ConvTranspose2d(1, 1, kernel_size=stride,padding=0,stride=stride, output_padding=0, groups=1, bias=False, dilation=1, padding_mode='zeros')
#         nn.init.constant_(self.pooldeconv.weight, 1)
#         self.weight = nn.Parameter(conv.weight)
#         self.bias = nn.Parameter(conv.bias)
#         self.stride = stride       
#         self.dilation = dilation 
#         self.padding = padding
#         self.kernel_size = kernel_size
       
#     def forward(self, input, selh=-torch.ones(1,1), selw=-torch.ones(1,1), mask=-torch.ones(1,1),stoch=True):
#         stride = self.stride
#         if stoch==False:
#             stride=1
#         batch_size, in_channels, in_h, in_w = input.shape
#         out_channels, in_channels, kh, kw =  self.weight.shape
#         afterconv_h = in_h-(kh-1)
#         afterconv_w = in_w-(kw-1)
#         out_h = int((afterconv_h+stride-1)/stride)
#         out_w = int((afterconv_w+stride-1)/stride)
        
#         unfold = torch.nn.Unfold(kernel_size=(kh, kw), dilation=self.dilation, padding=self.padding, stride=1)
#         inp_unf = unfold(input)
#         if stride!=1:
#             inp_unf = inp_unf.view(batch_size,in_channels*kh*kw,afterconv_h,afterconv_w)
#             if selh[0,0]==-1:
#                 resth = (out_h*stride)-afterconv_h
#                 restw = (out_w*stride)-afterconv_w
#                 selh = torch.cuda.LongTensor(out_h,out_w).random_(0, stride)
#                 selw = torch.cuda.LongTensor(out_h,out_w).random_(0, stride)
#                 #if resth!=0:
#                 #    selh[-1,:]=selh[-1,:]%(stride-resth);selh[:,-1]=selh[:,-1]%(stride-restw)
#                 #    selw[-1,:]=selw[-1,:]%(stride-resth);selw[:,-1]=selw[:,-1]%(stride-restw)
#                 #if mask[0,0]==-1
#                 #    mask = torch.ones(out_h,out_w,device=torch.device('cuda'))
#             rng_h = selh + torch.arange(0,out_h*stride,stride,device=torch.device('cuda')).view(-1,1)
#             rng_w = selw + torch.arange(0,out_w*stride,stride,device=torch.device('cuda'))
#             if mask[0,0]==-1:
#                 inp_unf = inp_unf[:,:,rng_h,rng_w].view(batch_size,in_channels*kh*kw,-1)
#             else:
#                 inp_unf = inp_unf[:,:,rng_h[mask>0],rng_w[mask>0]]

#         if self.bias is None:
#             out_unf = inp_unf.transpose(1, 2).matmul(self.weight.view(self.weight.size(0), -1).t()).transpose(1, 2)
#         else:
#             out_unf = (inp_unf.transpose(1, 2).matmul(self.weight.view(self.weight.size(0), -1).t()) + self.bias).transpose(1, 2)
    
#         if stride==1 or mask[0,0]==-1:
#             out = out_unf.view(batch_size,out_channels,out_h,out_w)
#             if stoch==False:
#                 out = F.avg_pool2d(out,self.stride,ceil_mode=True)
#         else:
#             out = torch.zeros(batch_size, out_channels,out_h,out_w,device=torch.device('cuda'))
#             out[:,:,mask>0] = out_unf
#         return out        

#     def forward_(self, input, selh=-torch.ones(1,1), selw=-torch.ones(1,1), mask=-torch.ones(1,1),stoch=True):
#         stride = self.stride
#         if stoch==False:
#             stride=1
#         batch_size, in_channels, in_h, in_w = input.shape
#         out_channels, in_channels, kh, kw =  self.weight.shape
#         afterconv_h = in_h-(kh-1)
#         afterconv_w = in_w-(kw-1)
#         out_h = (afterconv_h+stride-1)/stride
#         out_w = (afterconv_w+stride-1)/stride
        
#         unfold = torch.nn.Unfold(kernel_size=(kh, kw), dilation=self.dilation, padding=self.padding, stride=1)
#         inp_unf = unfold(input)
        
#         if self.bias is None:
#             out_unf = inp_unf.transpose(1, 2).matmul(self.weight.view(self.weight.size(0), -1).t()).transpose(1, 2)
#         else:
#             out_unf = (inp_unf.transpose(1, 2).matmul(self.weight.view(self.weight.size(0), -1).t()) + self.bias).transpose(1, 2)
    
#         out = out_unf.view(batch_size,out_channels,afterconv_h,afterconv_w)
#         if stoch==False:
#             out = F.avg_pool2d(out,self.stride,ceil_mode=True)
#         return out        

#     def sample(self,h,w,mask):
#         out_channels, in_channels, kh, kw =  self.weight.shape      
#         afterconv_h = h-(kh-1)
#         afterconv_w = w-(kw-1)
#         out_h = (afterconv_h+self.stride-1)/self.stride
#         out_w = (afterconv_w+self.stride-1)/self.stride
#         selh = torch.cuda.LongTensor(out_h,out_w).random_(0, self.stride)
#         selw = torch.cuda.LongTensor(out_h,out_w).random_(0, self.stride)
#         resth = (out_h*self.stride)-afterconv_h
#         restw = (out_w*self.stride)-afterconv_w
#         #print(resth)
#         #if resth!=0:
#         #    selh[-1,:]=selh[-1,:]%(self.stride-resth);selh[:,-1]=selh[:,-1]%(self.stride-restw)
#         #    selw[-1,:]=selw[-1,:]%(self.stride-resth);selw[:,-1]=selw[:,-1]%(self.stride-restw)
#         maskh = (out_h)*self.stride#-resth#+self.kernel_size-1
#         maskw = (out_w)*self.stride#-restw#+self.kernel_size-1   
#         rng_h = selh + torch.arange(0,out_h*self.stride,self.stride,device=torch.device('cuda')).view(-1,1)
#         rng_w = selw + torch.arange(0,out_w*self.stride,self.stride,device=torch.device('cuda'))
#         nmask = torch.zeros((maskh,maskw),device=torch.device('cuda'))
#         nmask[rng_h,rng_w] = 1
#         #rmask = mask * nmask
#         dmask = self.pooldeconv(mask.float().view(1,1,mask.shape[0],mask.shape[1]))
#         rmask = nmask * dmask
#         #rmask = rmask[:,:,:out_h,:out_w]
#         fmask = self.deconv(rmask)
#         fmask = fmask[0,0]
#         return selh,selw,fmask.long()

#     def get_size(self,h,w):
#         newh=(h-(self.kernel_size-1)+(self.stride-1))/self.stride
#         neww=(w-(self.kernel_size-1)+(self.stride-1))/self.stride
#         return newh,neww


# def savg_pool2d(x,size,ceil_mode=False):
#     b,c,h,w = x.shape
#     selh = torch.LongTensor(h/size,w/size).random_(0, size)
#     rngh = torch.arange(0,h,size).long().view(h/size,1).repeat(1,w/size).view(h/size,w/size)
#     selx = (selh+rngh).repeat(b,c,1,1)

#     selw = torch.LongTensor(h/size,w/size).random_(0, size)
#     rngw = torch.arange(0,w,size).long().view(1,h/size).repeat(h/size,1).view(h/size,w/size)
#     sely = (selw+rngw).repeat(b,c,1,1)
#     bv, cv ,hv, wv = torch.meshgrid([torch.arange(0,b), torch.arange(0,c),torch.arange(0,h/size),torch.arange(0,w/size)])
#     #x=x.view(b,c,h*w)
#     newx = x[bv,cv, selx, sely]
#     #ghdh
#     return newx

# def savg_pool2d_(x,size,ceil_mode=False):
#     b,c,h,w = x.shape
#     selh = torch.cuda.LongTensor(h/size,w/size).random_(0, size)
#     rngh = torch.arange(0,h,size,device=torch.device('cuda')).view(-1,1)
#     selx = selh+rngh

#     selw = torch.cuda.LongTensor(h/size,w/size).random_(0, size)
#     rngw = torch.arange(0,w,size,device=torch.device('cuda'))
#     sely = selw+rngw

#     #bv, cv ,hv, wv = torch.meshgrid([torch.arange(0,b), torch.arange(0,c),torch.arange(0,h/size),torch.arange(0,w/size)])
#     #x=x.view(b,c,h*w)
#     newx = x[:,:, selx, sely]
#     #ghdh
#     return newx
