import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from lib.models import ConvNet
from lib.train import train
from lib.BB import BB
from lib.AdaHessian import AdaHessian
import warnings
warnings.filterwarnings("ignore")
device = 'cuda:1'


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
     
cifar_trainset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
cifar_testset = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)

train_dataloader = torch.utils.data.DataLoader(cifar_trainset, batch_size=256, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(cifar_testset, batch_size=256, shuffle=False)

optims =  ['Adam', 'SGD_lr=0.1', 'SGD_lr=0.01',
           'AdaHessian', 'LBFGS', 'SGD momentum', 'BB']
          
def init_optim(optim, model):
    if optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = 5e-4)
    elif optim == 'SGD_lr=0.1':
        optimizer = torch.optim.SGD(model.parameters(), lr = 1e-1)
    elif optim == 'SGD_lr=0.01':
        optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2)
    elif optim == 'SGD momentum':
        optimizer = torch.optim.SGD(model.parameters(), momentum = 0.9, lr = 1e-2)
    elif optim == 'LBFGS':
        optimizer = torch.optim.LBFGS(model.parameters(), lr = 1e-2)
    elif optim == 'AdaHessian':
        optimizer = AdaHessian(model.parameters(), lr = 1e-1)
    elif optim == 'BB':
        optimizer = BB(model.parameters(), lr = 5e-3)
    return optimizer
    
results = {}
criterion = nn.CrossEntropyLoss()
for optim in optims:
    model = ConvNet()
    model.to(device)
    optimizer = init_optim(optim, model)
    res = train(model, optimizer, criterion, train_dataloader, 
            valid_dataloader, device, optim=optim, epochs=40, verbose=True)
    results[optim] = res
