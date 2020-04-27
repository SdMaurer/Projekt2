# Imports here
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

def load_image_data(data_dir): 
    
    data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

# Pass transforms in here, then run the next cell to see how the transforms look
    image_datasets = datasets.ImageFolder(train_dir, transform = data_transforms)

    test_datasets  = datasets.ImageFolder(test_dir , transform = test_transforms)

    valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)



# TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=32, shuffle=False)
    testloaders = torch.utils.data.DataLoader(test_datasets , batch_size=32)
    validloaders= torch.utils.data.DataLoader(valid_datasets, batch_size=32, shuffle=False)

    return dataloaders, validloaders, testloaders, image_datasets

