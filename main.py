from resnet import se_resnet_18
from data import CloudDataset, ToTensor
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import datetime
from torch.optim import lr_scheduler
from config import Config
from train import Trainer

img_dir = './data/images224'
labels_dir = './data/masks224/'
epochs = 100

cfig = Config() #config for the model
net = se_resnet_18()  #create CNN model.
criterion = nn.BCELoss()  #define the loss
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  #select the optimizer
#lr would divide gamma for every step_size,if you  schedult it by the class lr_scheduler.
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
# create the train_dataset_loader and val_dataset_loader.
train_tarnsformed_dataset = CloudDataset(img_dir=img_dir,labels_dir=labels_dir,transform=transforms.Compose([ToTensor()]))
val_tarnsformed_dataset = CloudDataset(img_dir=img_dir,labels_dir=labels_dir,val=True,transform=transforms.Compose([ToTensor()]))
train_dataloader = DataLoader(train_tarnsformed_dataset, batch_size=8, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_tarnsformed_dataset, batch_size=8, shuffle=True, num_workers=4)
trainer = Trainer('training', optimizer,exp_lr_scheduler, net, cfig, './log')
trainer.load_weights(trainer.find_last()) #加载最新的模型，并基于此模型继续训练。
trainer.train(train_dataloader, val_dataloader, criterion, epochs)
print('Finished Training')
