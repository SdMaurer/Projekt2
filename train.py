#import
import numpy as np
import torch
from torch import nn as nn
from torch import optim as optim
import nnModel
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from collections import OrderedDict
import json
from torch.autograd import Variable
import argparse
import os
import DataLoader

parser = argparse.ArgumentParser(description='Train a new network on a data set with transfer learning')


parser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory')
parser.add_argument('--gpu', type=bool, default='True', help='True: gpu, False: cpu')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epochs', type=int, default=5, help='num of epochs')
parser.add_argument('--arch', type=str, default='vgg16', help='architecture')
parser.add_argument('--batch_size', type=int, default=32,help='bacht size')
parser.add_argument('--hidden_units', type=int, default=1024, help='hidden units for layer')
parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='save train model to a file')
args = parser.parse_args()


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'





# Collect the arguments
args = parser.parse_args()
data_directory = args.data_dir
save_directory = args.save_dir
arch = args.arch
learning_rate = args.lr
hidden_units = args.hidden_units
epochs = args.epochs
batch_size = args.batch_size
gpu = args.gpu

dataloaders, validloaders, testloaders, image_datasets = DataLoader.load_image_data(data_dir)

# Create the model. Returns 0 if model cant be created
model = nnModel.create_model(arch, hidden_units)

# If we sucessfully create a model continue with the training
if model != 0:
    # Define the loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate) 

    # Train the model with validation
    trained_model=nnModel.train_model(model, dataloaders, validloaders,testloaders, criterion, optimizer, epochs, gpu)

    # Save the model
    nnModel.save_model(trained_model, image_datasets, learning_rate, batch_size, epochs, criterion, optimizer, hidden_units, arch)




