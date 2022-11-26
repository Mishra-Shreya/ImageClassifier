# For the completion of this project reference have been made from various codes examples available online including the pytorch documentation tutorials. 
# References: 
# 1. https://github.com/fotisk07/Image-Classifier
# 2. https://katba-caroline.com/what-flower-is-this-developing-an-image-classifier-with-deep-learning-using-pytorch/
# 3. https://github.com/ErkanHatipoglu/AIPND_final_project_part_2
# 4. https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#load-data
# 5. https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from collections import OrderedDict
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

import json
from workspace_utils import active_session

from PIL import Image
# from __future__ import print_function, division
plt.ion()   # interactive mode


#can use this code to change to CPU if Cuda is not available
##device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Check cuda
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
# print(device) 

structures = {"vgg13":25088,
              "vgg16":25088,
              "vgg19":25088}


def my_model(structure='vgg16', dropout=0.2,
             hidden_layer=1024,lr = 0.001):
    
    
    if structure == "vgg13":
        model = models.vgg13(pretrained=True)
    elif structure == "vgg16":
        model = models.vgg16(pretrained=True)
    elif structure == "vgg19":
        model = models.vgg19(pretrained=True)
    else:
        print("wrong model, try vgg13, vgg16, or vgg19")
        
    model = models.__dict__[structure](pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(structures[structure], hidden_layer)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(dropout)),
                          ('fc2', nn.Linear(hidden_layer, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        model.classifier = classifier
        
        criterion = nn.NLLLoss()
        
        optimizer = optim.Adam(model.classifier.parameters(), lr)
        
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        model = model.to(device)
        
        return model, optimizer, criterion, exp_lr_scheduler
    

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_model(checkpoint_path):
    chkpt = torch.load(checkpoint_path)
    structure = chkpt['structure']
    hidden_layer = chkpt['hidden_layer']
    learning_rate = chkpt['learning_rate']
    dropout = chkpt['dropout']
    model,_,_,_ = my_model(structure, dropout, hidden_layer, learning_rate)
    model.class_to_idx = chkpt['class_to_idx']
    model.load_state_dict(chkpt['state_dict'])
    
    
    return model
