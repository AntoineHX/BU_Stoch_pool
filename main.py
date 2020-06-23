'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse

from models import *
# from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch', default=128, type=int, help='batch_size')
parser.add_argument('--epochs', '-ep', default=10, type=int, help='epochs')
parser.add_argument('--scheduler', '-sc', dest='scheduler', default='',
                    help='cosine/multiStep/exponential')
parser.add_argument('--warmup_mul', '-wm', dest='warmup_mul', type=float, default=0, #2 #+ batch_size => + mutliplier
                    help='Warmup multiplier')
parser.add_argument('--warmup_ep', '-we', dest='warmup_ep', type=int, default=5,
                    help='Warmup epochs')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--stoch', '-s', action='store_true',
                    help='use stochastic pooling')
parser.add_argument('--network', '-n', dest='net', default='MyLeNetNormal',
                    help='Network')
parser.add_argument('--res_folder', '-rf', dest='res_folder', default='res/',
                    help='Results destination')
parser.add_argument('--postfix', '-pf', dest='postfix', default='',
                    help='Results postfix')    
parser.add_argument('--dataset', '-d', dest='dataset', default='CIFAR10',
                    help='Dataset')                  
args = parser.parse_args()
print(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
checkpoint=False

# Data
print('==> Preparing data..')
dataroot="~/scratch/data" #"./data"
download_data=False
transform_train = [
    # transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
]

transform_test = [
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
]

# trainset = torchvision.datasets.CIFAR10(
#     root=dataroot, train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(
#     trainset, batch_size=args.batch, shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(
#     root=dataroot, train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(
#     testset, batch_size=args.batch, shuffle=False, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat', 'deer',
#            'dog', 'frog', 'horse', 'ship', 'truck')

if args.dataset == 'CIFAR10': #(32x32 RGB)
    transform_train=transform_train+[transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    transform_test=transform_test+[transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]

    trainset = torchvision.datasets.CIFAR10(dataroot, train=True, download=download_data, transform=transforms.Compose(transform_train))
    # data_val = torchvision.datasets.CIFAR10(dataroot, train=True, download=download_data, transform=transforms.Compose(transform))
    testset = torchvision.datasets.CIFAR10(dataroot, train=False, download=download_data, transform=transforms.Compose(transform_test))
elif args.dataset == 'TinyImageNet': #(Train:100k, Val:5k, Test:5k) (64x64 RGB)
    transform_train=transform_train+[transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    transform_test=transform_test+[transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    
    trainset = torchvision.datasets.ImageFolder(os.path.join(dataroot, 'tiny-imagenet-200/train'), transform=transforms.Compose(transform_train))
    # data_val = torchvision.datasets.ImageFolder(os.path.join(dataroot, 'tiny-imagenet-200/val'), transform=transforms.Compose(transform))
    testset = torchvision.datasets.ImageFolder(os.path.join(dataroot, 'tiny-imagenet-200/test'), transform=transforms.Compose(transform_test))
else:
    raise Exception('Unknown dataset')
    
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
#normal cuda convolution
# net = MyLeNetNormal() #11.3s - 49.4%  #2.3GB

#strided convolutions instead of pooling
#net = MyLeNetStride() #5.7s - 41.45% (5 epochs) #0.86GB

#convolution with matrices unfold
#net = MyLeNetMatNormal() #19.6s - 41.3%  #1.7GB 

#stochastic like fig.2 paper
#net = MyLeNetMatStoch() # 16.8s - 41.3%  #1.8GB

#storchastic Bottom-UP like fig.3 paper
# net = MyLeNetMatStochBU() # 10.5s - 45.3%  #1.3GB

net=globals()[args.net]()
#print(net)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

log = []
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    print('WARNING : Log & Lr-Scheduler resuming is not available')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)

# Training
max_grad = 1 #Max gradient value #Limite catastrophic drop
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=max_grad, norm_type=2) #Prevent exploding grad with RNN
        optimizer.step()

        #Log
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
    # if args.net in {'MyLeNetMatNormal', 'MyLeNetMatStoch', 'MyLeNetMatStochBU'}:
    #     print('Comp',net.comp)
    return train_loss/(batch_idx+1), 100.*correct/total

#determinisitc test
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs,stoch=False)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        if checkpoint:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

    return test_loss/(batch_idx+1), acc

#Stochastic test
def stest(epoch,times=10):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            out = torch.zeros(times,inputs.shape[0],10).cuda()
            for l in range(times):
                out[l] = net(inputs,stoch=True)
            outputs = out.mean(0)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

import matplotlib.pyplot as plt
def plot_res(log, best_acc,fig_name='res'):
    """Save a visual graph of the logs.

        Args:
            log (dict): Logs of the training generated by most of train_utils.
            fig_name (string): Relative path where to save the graph. (default: res)
    """
    epochs = [x["epoch"] for x in log]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(30, 15))

    ax[0].set_title('Loss')
    ax[0].plot(epochs,[x["train_loss"] for x in log], label='Train')
    ax[0].plot(epochs,[x["test_loss"] for x in log], label='Test')
    ax[0].legend()
        
    ax[1].set_title('Acc %s'%best_acc)
    ax[1].plot(epochs,[x["train_acc"] for x in log], label='Train')
    ax[1].plot(epochs,[x["test_acc"] for x in log], label='Test')
    ax[1].legend()
     

    fig_name = fig_name.replace('.',',').replace(',,/','../')
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close()

#from warmup_scheduler import GradualWarmupScheduler
def get_scheduler(schedule, epochs, warmup_mul, warmup_ep):
    scheduler=None
    if schedule=='cosine':
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.)
    elif schedule=='multiStep':
        #Multistep milestones inspired by AutoAugment
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, 
            milestones=[int(epochs/3), int(epochs*2/3), int(epochs*2.7/3)], 
            gamma=0.1)
    elif schedule=='exponential':
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (1 - epoch / epochs) ** 0.9)
    elif not(schedule is None or schedule==''):
        raise ValueError("Lr scheduler unknown : %s"%schedule)

    #Warmup
    if warmup_mul>=1:
        scheduler=GradualWarmupScheduler(optimizer, 
            multiplier=warmup_mul, 
            total_epoch=warmup_ep, 
            after_scheduler=scheduler)
    
    return scheduler

### MAIN ###
print_freq=args.epochs/10
res_folder=args.res_folder
filename = ("{}-{}epochs".format(args.net,start_epoch+args.epochs))+args.postfix
log = []

#Lr-Scheduler
scheduler=get_scheduler(args.scheduler, args.epochs, args.warmup_mul, args.warmup_ep)

print('==> Training model..')
t0 = time.perf_counter()
for epoch in range(start_epoch, start_epoch+args.epochs):
    
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)

    if scheduler is not None:
        scheduler.step()

    #### Log ####
    log.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
    })

    ### Print ###
    if(print_freq and epoch%print_freq==0):
        print('-'*9)
        print('\nEpoch: %d' % epoch)
        print("Acc : %.2f / %.2f"%(train_acc, test_acc))
        print("Loss : %.2f / %.2f"%(train_loss, test_loss))
        print('Time:',time.perf_counter() - t0)

exec_time=time.perf_counter() - t0
print('-'*9)
print('Best Acc : %.2f'%best_acc)
print('Training time (min):',exec_time/60)


import json
try:
    with open(res_folder+"log/%s.json" % filename, "w+") as f:
        json.dump(log, f, indent=True)
        print('Log :\"',f.name, '\" saved !')
except:
    print("Failed to save logs :",filename)
    print(sys.exc_info()[1])
try:
    plot_res(log,best_acc, fig_name=res_folder+filename)
    print('Plot :\"',res_folder+filename, '\" saved !')
except:
    print("Failed to plot res")
    print(sys.exc_info()[1])
