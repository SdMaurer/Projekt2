import torch
from torch import nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import models
import torch.nn.functional as F
import time
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import DataLoader


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
   


def create_model(arch, hidden_units):
    '''
        Creates a pretrained model using VGG19 or Densenet161 and returns the model
        
        Inputs:
        arch - The architecture to be used. Either 'vgg19' or 'densenet161'
        hidden_units - The number of units in the hidden layer
        
        Outputs:
        model - The created (loaded) pretrained model
    '''
    # Load a pretrained network (vgg19 or densenet161)
    # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    print("Creating the model...")
    #images, labels = next(iter(DataLoader))

   # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   # images.to('cuda'), labels.to('cuda')
    model = models.vgg16(pretrained=True)
    
    # Turn off gradient
    for param in model.parameters():
        param.requires_grad = False
#Define our new classifier    
    model.classifier = nn.Sequential(nn.Linear(25088, 1024),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(1024, 102),
                                 nn.LogSoftmax(dim=1))
    
    #Set default tensor type, then move model and data to GPU
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model.to(device)
   # images, labels = images.to(device), labels.to(device)
#Setup loss function and optimizer
    criterion = nn.NLLLoss()
    #optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    #Take a pass through the model
    #output = model.forward(images)
    #loss = criterion(output, labels)
    #loss.backward()
   # optimizer.step()
    print('1 Pass Train Complete')

    #optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    #model.to(device);



    print("Done creating the model\n")
    return model

def train_model(model, dataloaders, validloaders, testloaders, criterion, optimizer, epochs, use_gpu):
    print("Training the model...\n")
    
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    model.to(device)
    epochs = 5
    steps = 0
    running_loss = 0
    print_every = 20


    for epoch in range (epochs):
                
        
        for images,labels in (dataloaders):
            steps+=1
            
        # Move input and label tensors to the default device
            images, labels = images.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                test_loss=0
                accuracy=0
                model.eval()
                
            
                with torch.no_grad():
                    for images, labels in testloaders:
                        images, labels = images.to(device), labels.to(device)
                        logps = model.forward(images)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()   #keep track of test los
                    
                    # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)  ##first largest value in our probability
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
            ''' print(f"Epoch {epoch+1}/{epochs}.. 
                   f"Train loss: {running_loss/print_every:.3f}.. "
                   f"Test loss: {test_loss/len(testloaders):.3f}.. "
                   f"Test accuracy: {accuracy/len(testloaders):.3f}")   '''        
                    
          
            
    running_loss = 0
    model.train()
    print("\nDone training the model \n")
    return model 
        
        
        
def save_model(model, image_datasets, learning_rate, batch_size, epochs, criterion, optimizer, hidden_units, arch):  
    
    print("Saving the model...")
    
    #save the image dataset
    model.class_to_idx = image_datasets.class_to_idx
    
    checkpoint = {'state_dict': model.state_dict(),
              
                  'arch': 'vgg16',
              
                  'input_size':25088,
              
                  'output_size':102,
              
                #  'hidden_layer_size': 1024,
              
                  'learning_rate': 0.001,
                  
                  ' classifier' : model.classifier,

                  'epochs': 5,

                  'class_to_idx': model.class_to_idx, 

                  'optimizer.state_dict': optimizer.state_dict(),
                 
                  'state_dict': model.state_dict()}
    
    torch.save(checkpoint, 'filepath')
    print("Done saving the model")
    

    
def load_model(filepath):
    '''
        Loads a model using a checkpoint.pth file
        
        Inputs:
        checkpoint_file - The file path and name for the checkpoint
        
        Outputs:
        model - Returns the loaded model
    '''
    print("Loading the model...")
    checkpoint = torch.load(filepath)
    
    if checkpoint['arch'] == 'vgg16':
        model=models.vgg16(pretrained=True)
      
    model.classifier =checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    #model.class_to_idx = checkpoint['class_to_idx']
    
    for param in model.parameters():
        param.res_grad=False
    
         
    

    model.eval()
   # model.eval()
    print("Done loading the model")
    return model

from PIL import Image

import numpy as np

def process_image(image):


    
    #im = Image.open("flowers/test/1/image_06743.jpg")
    
    im = Image.open(image)

    width, height = im.size
    
   # print(np.shape(im))
    
    #fig,ax=plt.subplots()
    #ax.imshow(im)
    

    if width > height:
        
        ratio = width/height

        im.thumbnail((ratio*256,256))

    elif height > width:
    

       im.thumbnail((256,height/width*256))
        
        

    new_width, new_height = im.size # take the size of resized imagenew_width=224

    

    left = (new_width - 224)/2

    top  = (new_height - 224)/2

    right = (new_width + 224)/2

    bottom = (new_height+ 224)/2

    im=im.crop((left, top, right, bottom))

    np_image=np.array(im)

    np_image = np_image / 255

    means=np.array([0.485, 0.456, 0.406])

    std= np.array([0.229, 0.224, 0.225])

    np_image=(np_image-means)/std

    np_image = np_image.transpose((2,0,1))
    

    return torch.tensor(np_image)

def imshow(image, ax=None, title=None):
    
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    


    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
     

    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax






def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    
    
    # TODO: Implement the code to predict the class from an image file
       
    model.eval() # inference mode
    img = process_image(image_path)

    img = img.to(device)
    
    # Add batch of size 1 to image so the first argument is going to be a batch not a single image

    img = img.unsqueeze(0)  

    
    # Predict top 5
    
    with torch.no_grad():
        logits = model.forward(img)
        probs, probs_labels = torch.topk(logits, topk)
        probs = probs.exp() # calc all exponential of all elements
        class_to_idx = model.class_to_idx
    
     
    # Use Tensor.cpu() to copy the tensor to host memory first. Still not sure why but i get error without it
    
    probs = probs.cpu().numpy()
    probs_labels = probs_labels.cpu().numpy()
    
    # change indices to make them to a class
    classes_indexed = {model.class_to_idx[i]: i for i in model.class_to_idx}
    
    # Don't forget to convert it to a list
    classes_list = list()
    
    for label in probs_labels[0]:
        classes_list.append(cat_to_name[classes_indexed[label]])
        
    return (probs[0], classes_list)

def plot_solution(image_path, model):
    
    # Make prediction
    probs, classes = predict(image_path, model)
    #print(probs,classes)
    
    image = process_image(image_path)
    axs = imshow(image, ax = plt)
    #max_index = classes[0]
    #plt.title(max_index)
    index=image_path.split('/')[2]
    plt.title(cat_to_name[str(index)])

    
    
    plt.figure(figsize=(4,4))
    y_pos = np.arange(len(classes))
    
    performance = np.array(probs)
    
    plt.barh(y_pos, performance, align='center',
        color=sns.color_palette()[0])
    
    y_pos, classes
   # plt.yticks(y_pos, flower_names)
    plt.yticks(y_pos, classes)
    plt.gca().invert_yaxis()

