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


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_pil = Image.open(image)
   
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = adjustments(img_pil)
    
    return img_tensor
    

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    model.eval()
    model.cpu()
    with torch.no_grad():
        image = process_image(image_path)
        image = image.unsqueeze(0)
        output = model.forward(image)
        top_prob, top_labels = torch.topk(output, topk)
        top_prob = top_prob.exp()
        top_prob_array = top_prob.data.numpy()[0]

        inv_class_to_idx = {v: k for k, v in model.class_to_idx.items()}

        top_labels_data = top_labels.data.numpy()
        top_labels_list = top_labels_data[0].tolist()  

        top_classes = [inv_class_to_idx[x] for x in top_labels_list]
    
        return top_prob_array, top_classes

# Initialize
parser = argparse.ArgumentParser(description="This program predicts flowers' names from their images",
								 usage='''
        needs a saved checkpoint
        python predict.py ( use default image 'flowers/test/1/image_06743.jpg' and root directory for checkpoint)
        python predict.py /path/to/image checkpoint (predict the image in /path/to/image using checkpoint)
        python predict.py --top_k 3 (return top K most likely classes)
        python predict.py --category_names cat_to_name.json (use a mapping of categories to real names)
        python predict.py --gpu (use GPU for inference)''',
								 prog='predict')

## Get path of image
parser.add_argument('path_to_image', action="store", nargs='?', default='flowers/test/5/image_05159.jpg', help="path/to/image")
## Get path of checkpoint
# https://stackoverflow.com/questions/4480075/argparse-optional-positional-arguments
# Thanks to: Vinay Sajip 
parser.add_argument('path_to_checkpoint', action="store", nargs='?', default='classifier.pth', help="path/to/checkpoint")
## set top_k
parser.add_argument('--top_k', action="store", default=5, type=int, help="enter number of guesses", dest="top_k")
## Choose json file:
parser.add_argument('--category_names', action="store", default="cat_to_name.json", help="get json file", dest="category_names")
## Set GPU
parser.add_argument('--gpu', action="store_true", default=False, help="Select GPU", dest="gpu")

## Get the arguments
args = parser.parse_args()

arg_path_to_image =  args.path_to_image
arg_path_to_checkpoint = args.path_to_checkpoint
arg_top_k =  args.top_k
arg_category_names =  args.category_names
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

# Use GPU if it's selected by user and it is available
device = torch.device("cuda" if arg_gpu else "cpu")
print()
print('Will use {} for prediction...'.format(device))
print()

print()
print("Path of image: {} \nPath of checkpoint: {} \nTopk: {} \nCategory names: {} ".format(arg_path_to_image, arg_path_to_checkpoint, arg_top_k, arg_category_names))
print('GPU: ', arg_gpu)
print()

## Label mapping
print('Mapping from category label to category name...')
print()
with open(arg_category_names, 'r') as f:
    cat_to_name = json.load(f)
    
    
## Loading model
print('Loading model........................ ')
print()

my_model  = utils.load_model(arg_path_to_checkpoint)

my_model.eval()

# https://knowledge.udacity.com/questions/47967
# idx_to_class = {v:k for k, v in my_model.class_to_idx.items()}

# https://github.com/SeanNaren/deepspeech.pytorch/issues/290
# Thanks to @ XuesongYang
# used helper.py
# https://www.tutorialspoint.com/matplotlib/matplotlib_bar_plot.htm
# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.bar.html

print(arg_path_to_image)
probs, classes = predict('{}'.format(arg_path_to_image), my_model, arg_top_k)

print()
print('The model predicts this flower as: ')

for count in range(arg_top_k):
    print("{} with a probability of {:.3f} ".format(cat_to_name[classes[count]], probs[count]))
        
