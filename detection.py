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
import matplotlib.pyplot as plt

cfig = Config()
net = se_resnet_18()  #create CNN model.
criterion = nn.BCELoss()  #define the loss

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  #select the optimizer
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
# create the train_dataset_loader and val_dataset_loader.
cloud_data = CloudDataset(img_dir='./data/images224', labels_dir='./data/masks224/')
trainer = Trainer('inference', optimizer, exp_lr_scheduler, net, cfig, './log')
trainer.load_weights(trainer.find_last())

for x in range(1050,2000,50):#600
	images = cloud_data[x]['image']
	gt_map = cloud_data[x]['gt_map']
	mask = trainer.detect(images)
	print(mask.shape)
	plt.subplot(131)
	plt.imshow(images)
	plt.subplot(132)
	plt.imshow(gt_map)
	plt.subplot(133)
	plt.imshow(mask)
	plt.show()
	input()
