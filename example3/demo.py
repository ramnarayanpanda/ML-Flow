import mlflow 
import argparse 
import os 
import numpy as np 
import pandas as pd 


import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR 
from torchvision import datasets, transforms 
import mlflow.pytorch



class Config:
    EPOCHS = 10 
    BATCH_SIZE = 32 
    LR = 0.01 
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
    GAMMA = 0.7 
    SEED = 42 
    LOG_INTERVAL = 10   # how many intervals into, you want to print the outcome 
    TEST_BATCH_SIZE = 1000 
    DRY_RUN = True 
    

config = Config() 


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__() 
        self.conv1 = nn.Conv2d(1, 32, 3, 1) 
        self.conv2 = nn.Conv2d(32, 64, 3, 1) 
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5) 
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x) 
        x = self.conv2(x) 
        x = F.relu(x) 
        x = F.max_pool2d(x, 2)
        
        x = self.dropout1(x)
        x = torch.flatten(x, 1) 
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.dropout2(x) 
        x = self.fc2(x)
        
        output = F.log_softmax(x, dim=1) 
        return output 
    
    
def train(config, model, device, train_loader, optimizer, epoch):
    
    model.train() 
    
    # print('\n\n\nfine till here1')
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = F.cross_entropy(pred, target) 
        loss.backward()
        optimizer.step() 
        
        if batch_idx % config.LOG_INTERVAL == 0:
            print(f"train epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100.0 *batch_idx / len(train_loader):.0f})]\t Loss:  {loss.item():.6f}")
            
            # dry run means of you are testing it for 1 time if your code works or not
            # then the for loop will be broken after just one batch
            if config.DRY_RUN:
                break       
    
    
def test(model, device, test_loader):
    pass  


torch.manual_seed(config.SEED) 
train_kwargs = {'batch_size': config.BATCH_SIZE}
test_kwargs = {'batch_size': config.TEST_BATCH_SIZE}

if config.DEVICE == 'cuda':
    cuda_kwargs = {'num_workers':1, 'pin_memory':True, 'shuffle':True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
    

transforms = transforms.Compose(
    [transforms.ToTensor()]
)


train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transforms)
test_dataset = datasets.MNIST('../data', train=False, transform=transforms)


train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs) 
test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs) 


model = ConvNet().to(config.DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=config.LR) 
scheduler = StepLR(optimizer, step_size=1, gamma=config.GAMMA)

# print('\n\n\nfine till here2')

# train loop 
for epoch in range(1, config.EPOCHS + 1):
    train(config, model, config.DEVICE, train_loader, optimizer, epoch)
    scheduler.step() 
    
    
with mlflow.start_run() as run:
    mlflow.pytorch.log_model(model, 'model') 
    model_path = mlflow.get_artifact_uri('model') 
    loaded_torch_model = mlflow.pytorch.load_model(model_path) 
    
    model.eval() 
    with torch.no_grad():
        test_datapoints, test_target = next(iter(test_loader))
        pred = loaded_torch_model(test_datapoints[0].unsqueeze(0).to(config.DEVICE))
        actual = test_target[0].item()
        predicted = torch.argmax(pred).item()
        print(f"actual:{actual}, predicted:{predicted}")