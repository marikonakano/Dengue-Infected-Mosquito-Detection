# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:26:15 2023

Training the LSTM using MelSpec Data

(1) ReadJson file composed by 150 datas for each class (health / infected)
(2) Generate Dataloader for training and valicdation 
(3) Generate LSTM models (Bi-LSTM and LSTM) with two layers

This program can be run in GPU    

@author: Mariko Nakano
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import numpy as np
from torch import optim
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###### models ######

class lstm_model(nn.Module):
    def __init__(self):
        super().__init__()
        #### Bi-LSTM
        self.lstm = nn.LSTM(input_size=32, hidden_size=256, num_layers = 2, batch_first=True, dropout=0.2, bidirectional=True)
        self.linear = nn.Linear(512,2)
        ### LSTM  If you want to use LSTM model these two lines must be activated
        #self.lstm = nn.LSTM(input_size=32, hidden_size=256, num_layers = 2,  batch_first=True, dropout=0.2)
        #self.linear = nn.Linear(256,2)
        
    def forward(self, x):
        
        x, _ = self.lstm(x)  # Get output
        x = self.linear(x[:,-1,:])  # flatten
        
        return x
    
###### DataLoader  ######

### Lectura de json data ###
json_path = '../Dataset/Melspec.json'

##### Guardar los pesos de la red #####
save_ruta ='../Modelo/prueba.pth'

with open(json_path, 'r') as f:
  data = json.load(f)

### Dataset ###  
from torch.utils.data import Dataset

class CreateDataset(Dataset):
    def __init__(self, X, y):
        # convert into PyTorch tensors and remember them
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y)

    def __len__(self):
        # this should return the size of the dataset
        return len(self.X)

    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features = self.X[idx]
        target = self.y[idx]
        return features, target
    
 
train_size = 130*2   #  130 data/class for training
test_size = 20*2    #  20 data/class for training
batch_size= 16

X_train, X_test, y_train, y_test = train_test_split(data['mspec'], data['labels'], test_size=test_size, random_state=42, stratify=data['labels'])


train_data = CreateDataset(X_train,y_train)
test_data  = CreateDataset(X_test,y_test)

Trainloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
Testloader = DataLoader(test_data, batch_size=batch_size)


###### create model ######
net = lstm_model()
print(net)
net.to(device)
# Cross Entropy Loss
loss_fnc = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(net.parameters(), lr=0.000005)

Num_epoch = 300

# Log for evaluation
record_loss_train = []
record_loss_test = []

#### Training ####
for epoch in tqdm(range(Num_epoch)):  # 
    net.train()  # Modo de entrenamiento
    loss_train = 0
    for j, (x, t) in enumerate(Trainloader):  # 
        x, t = x.to(device), t.to(device)  # Los datos envian a GPU
        y = net(x)
        loss = loss_fnc(y, t)
        loss_train += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_train /= j+1
    record_loss_train.append(loss_train)

#   Evaluacion
    net.eval()  # 
    loss_test = 0
    
    for j, (x, t) in enumerate(Testloader):  
        x, t = x.to(device), t.to(device)
        y = net(x)
        loss = loss_fnc(y, t)
        loss_test += loss.item()
    loss_test /= j+1
    record_loss_test.append(loss_test)

    if epoch%1 == 0:
        print("Epoch:", epoch, "Loss_Train:", loss_train, "Loss_Test:", loss_test)
        
    if epoch == 0:
        min_loss = loss_test
    else:
        print("loss minimo : ", min_loss)
        if loss_test < min_loss:
            print("El mejor modelo ha sido actualizado")   # Get best model
            torch.save(net.state_dict(), save_ruta)
            min_loss = loss_test


import matplotlib.pyplot as plt

plt.plot(range(len(record_loss_train)), record_loss_train, label="Train")
plt.plot(range(len(record_loss_test)), record_loss_test, label="Test")
plt.legend()

plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()


####### Evaluation without Monte Carlo Dropout
correct = 0
total = 0
net.load_state_dict(torch.load(save_ruta))  # Cargar el mejor modelo guardado

net.to(device)
net.eval()  # Evaluation mode
for i, (x, t) in enumerate(tqdm(Testloader)):
    x, t = x.to(device), t.to(device)  # GPU operation
    y = net(x)
    correct += (y.argmax(1) == t).sum().item()
    total += len(x)

print(" ")    
print("Accuracy:", str(np.round(correct/total*100, 4)) + "%")

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


num_params = get_n_params(net)
print("number of entrenable parameters = ", num_params)

