# For the completion of this project reference have been made from various codes examples available online including the pytorch documentation tutorials. 
# References: 
# 1. https://github.com/fotisk07/Image-Classifier
# 2. https://katba-caroline.com/what-flower-is-this-developing-an-image-classifier-with-deep-learning-using-pytorch/
# 3. https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#load-data
# 4. https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad

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

import utils

from PIL import Image
# from __future__ import print_function, division
plt.ion()   # interactive mode

## https://pymotw.com/3/argparse/
## https://docs.python.org/3/library/argparse.html#the-add-argument-method

## use argparse for command line inputs

# Initialize
parser = argparse.ArgumentParser(description='This is a model training program for a dateset of flowers using pytorch',
								 usage='''
        python train.py (data set shall be initially extracted to the 'flowers' directory)
        python train.py data_dir (data set shall be initially extracted to the 'data_dir' directory)
        python train.py data_dir --save_dir save_directory (set directory to save checkpoints)
        python train.py data_dir --arch "vgg13" (choose architecture from vgg13, vgg16 and vgg19)
        python train.py data_dir --learning_rate 0.01 --hidden_units [1024, 512, 256] --epochs 20 (set hyperparameters)''',
								 prog='train')


## Get dataset location, use flowers as default
# https://stackoverflow.com/questions/4480075/argparse-optional-positional-arguments
# Thanks to: Vinay Sajip 
parser.add_argument('data_directory', action="store", nargs='?', default="flowers", help="dataset directory")

## Set directory to save checkpoints
parser.add_argument('--save_dir', action="store", default="", help="saving directory for checkpoint", dest="save_directory")

## Choose architecture:
parser.add_argument('--arch', action="store", default="vgg16", choices=['vgg13', 'vgg16', 'vgg19'],
					 help="you can only choose vgg13, vgg16 or vgg19", dest="architecture")

## Set hyperparameters
parser.add_argument('--learning_rate', action="store", default=0.001, type=float, help="Set Learning rate",
					 dest="learning_rate")
parser.add_argument('--hidden_layers', action="store", default=1024, type=int, help="Set hidden layers",
					 dest="hidden_layers")
parser.add_argument('--epochs', action="store", default=15, type=int, help="set epochs", dest="epochs")

parser.add_argument('--dropout', action = "store", default = 0.2, dest = "dropout")
## Set GPU
parser.add_argument('--gpu', action="store_true", default=False, help="Select GPU", dest="gpu")

## Get the arguments
args = parser.parse_args()

arg_data_dir =  args.data_directory
arg_save_dir =  args.save_directory
arg_architecture =  args.architecture
arg_lr = args.learning_rate
arg_hidden_layers = args.hidden_layers
arg_epochs = args.epochs
arg_dropout = args.dropout
# Use GPU if it's selected by user and it is available
if args.gpu and torch.cuda.is_available(): 
	arg_gpu = args.gpu
# if GPU is selected but not available use CPU and warn user
elif args.gpu:
	arg_gpu = False
	print('GPU is not available, will use CPU...')
	print()
# Otherwise use CPU
else:
	arg_gpu = args.gpu

print()
print("Data directory: root/{}/ \nSave directory: root/{} \nArchitecture: {} ".format(arg_data_dir, arg_save_dir, arg_architecture))
print('Learning_rate: ', arg_lr)
print('Hidden units: ', arg_hidden_layers)
print('Epochs: ', arg_epochs)
print('Dropout: ', arg_dropout)
print('GPU: ', arg_gpu)
print()
   
    
# data_dir = 'flowers'
data_dir = arg_data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(35),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    # The validation and testing sets are used to measure the model's performance
    ## on data it hasn't seen yet. For this you don't want any scaling or
    ## rotation transformations,but you'll need to resize then crop the images to the appropriate size.
    
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]),
}

# TODO: Load the datasets with ImageFolder
# data_dir = 'flowers'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'valid', 'test']} 

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'valid','test']}


dataset_sizes = {x: len(image_datasets[x])
                 for x in ['train', 'valid', 'test']}

class_names = image_datasets['train'].classes

#can use this code to change to CPU if Cuda is not available
##device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Check cuda
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
# print(device) 


model,optimizer,criterion,exp_lr_scheduler = utils.my_model(structure=arg_architecture, 
                                                            dropout=arg_dropout, 
                                                            hidden_layer=arg_hidden_layers, 
                                                            lr=arg_lr)


def train_model(model, criterion, optimizer, scheduler,
                num_epochs=15, device = 'cuda'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

#Visualizing the model predictions
##Generic function to display predictions for a few images

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return model.train(mode=was_training)
                


with active_session():
    model = train_model(model, criterion, optimizer, 
                    exp_lr_scheduler, num_epochs=arg_epochs, device=device)
    
# visualize_model(model)
# plt.ioff()
# plt.show()
    
# TODO: Do validation on the test set
def check_accuracy_on_test(model, data, cuda=False):
    model.eval()
    model.to(device='cuda') 
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in (dataloaders[data]):
            images, labels = data
            #images, labels = images.to('cuda'), labels.to('cuda')
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    
check_accuracy_on_test(model, 'test', True)


# TODO: Save the checkpoint 

# https://thispointer.com/how-to-create-a-directory-in-python/
# https://stackoverflow.com/questions/9573244/how-to-check-if-the-string-is-empty
# Thanks to @Andrew Clark
if arg_save_dir:
	if not os.path.exists(arg_save_dir):
		os.mkdir(arg_save_dir)
		print("Directory " , arg_save_dir ,  " has been created for saving checkpoints")
	else:
		print("Directory " , arg_save_dir ,  " allready exists for saving checkpoints")
	save_dir = arg_save_dir + '/classifier.pth'
else:
	save_dir = 'classifier.pth'

# Save the checkpoint 
model.class_to_idx = image_datasets['train'].class_to_idx
model.cpu()
torch.save({'structure': arg_architecture,
            'hidden_layer': arg_hidden_layers,
            'learning_rate': arg_lr,
            'epochs': arg_epochs,
            'dropout': arg_dropout,
            'state_dict': model.state_dict(), 
            'class_to_idx': model.class_to_idx}, 
            save_dir)

model = utils.load_model(save_dir)
print('Saved model:')
print(model)
