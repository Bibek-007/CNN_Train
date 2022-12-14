#!/usr/bin/env python
# coding: utf-8

# # PyTorch Assignment: Convolutional Neural Network (CNN)

# **[Duke Community Standard](http://integrity.duke.edu/standard.html): By typing your name below, you are certifying that you have adhered to the Duke Community Standard in completing this assignment.**
# 
# Name: Bibekananda Bachhar

# ### Convolutional Neural Network
# 
# Adapt the CNN example for MNIST digit classfication from Notebook 3A. 
# Feel free to play around with the model architecture and see how the training time/performance changes, but to begin, try the following:
# 
# Image ->  
# convolution (32 3x3 filters) -> nonlinearity (ReLU) ->  
# convolution (32 3x3 filters) -> nonlinearity (ReLU) -> (2x2 max pool) ->  
# convolution (64 3x3 filters) -> nonlinearity (ReLU) ->  
# convolution (64 3x3 filters) -> nonlinearity (ReLU) -> (2x2 max pool) -> flatten ->
# fully connected (256 hidden units) -> nonlinearity (ReLU) ->  
# fully connected (10 hidden units) -> softmax 



###

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
import numpy as np
from torchvision import datasets,transforms
import torch.optim as optim

trainset = datasets.MNIST(download=True,train=True,root='./dataset',transform=transforms.ToTensor())
testset = datasets.MNIST(download=True,train=False,root='./dataset',transform=transforms.ToTensor())

trainloader = torch.utils.data.DataLoader(trainset,batch_size=100,shuffle=True)
testloader = torch.utils.data.DataLoader(testset,batch_size=100,shuffle=False)

class CNN_NEW(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size=3)
        self.conv2 = nn.Conv2d(32,32,kernel_size=3)
        self.conv3 = nn.Conv2d(32,64,kernel_size=3)
        self.conv4 = nn.Conv2d(64,64,kernel_size=3)
        self.fc1 = nn.Linear(20736,256)
        self.fc2 = nn.Linear(256,10)

    def forward(self,X):
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        max_pool = nn.MaxPool2d(2,1)
        X = max_pool(X)
        X = F.relu(self.conv3(X))
        X = F.relu(self.conv4(X))
        X = max_pool(X)
       #print(X.flatten().shape)
        X = self.fc1(X.view(100,20736))
        X = self.fc2(X)
        #print(X.shape)
        #return F.softmax(X,dim=1)
        return X

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('running on gpu')
else:
    device = torch.device('cpu')
    print('running on cpu')
    
mymodel = CNN_NEW().to(device)

BATCH_SIZE=100
EPOCHS=3

optimizer = optim.Adam(mymodel.parameters(),lr=0.001)
Loss = nn.CrossEntropyLoss()

batches = iter(trainloader)

for e in range(EPOCHS):
    for batch in tqdm(batches):
        mymodel.zero_grad()
        image, label = batch
        image = image.to(device)
        label = label.to(device)
        #print(image.shape)

        Y = mymodel(image)
        #print(Y.shape,label.shape)
        loss = Loss(Y,label)
        loss.backward()
        optimizer.step()
        batches = iter(trainloader)

tests = iter(testloader)
correct = 0
total = 0

with torch.no_grad():
     for batch in tqdm(tests):
        img, lbl = batch
        imag = img.to(device)
        lbl = lbl.to(device)
        Y = mymodel(imag)
        predict = torch.argmax(Y,dim=1)
    
#   plt.imshow(img[1].view(28,28))
#    plt.title(Y)

        for p,l in zip(predict,lbl):
             #print(predict,lbl)
            if p == l:
                correct+=1
                total+=1

print("Accuracy: {}%".format((correct/total)*100))
#print(correct,total)






