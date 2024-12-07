# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:26:15 2023

Entrenar la red LSTM usando datos de MelSpec


(1) Leer archivo de Json 150 datos por clase (sano/infectado)
(2) Generar DataLoader de entrenamiento y prueba
(3) Generar modelo de LSTM con una capa
    Variación-1: El número de capas de LSTM =2, Hidden =128, Dropout=0.2, Unidireccional
                 Agregar Dropout dentro de la capas de LSTM

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

###### modelo ######

class lstm_model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=32, hidden_size=256, num_layers = 2, batch_first=True, dropout=0.5, bidirectional=True)
        self.linear = nn.Linear(512,2)
        #self.lstm = nn.LSTM(input_size=32, hidden_size=256, num_layers = 2,  batch_first=True, dropout=0.2)
        #self.linear = nn.Linear(256,2)
        
    def forward(self, x):
        
        x, _ = self.lstm(x)  # solo salida, no usa estados internos
        x = self.linear(x[:,-1,:])  # realizando flatten.
        
        return x
    
###### Construcción de Dataloader ######

### Lectura de json data ###
json_path = '../Dataset/Json/Melspec.json'

##### Guardar los pesos de la red #####
save_ruta ='../Modelo/prueba.pth'

with open(json_path, 'r') as f:
  data = json.load(f)

### Separación de datos en "Entrenamiento" y "Prueba" ###  
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
    
 
train_size = 130*2
test_size = 20*2
batch_size= 16

X_train, X_test, y_train, y_test = train_test_split(data['mspec'], data['labels'], test_size=test_size, random_state=42, stratify=data['labels'])


train_data = CreateDataset(X_train,y_train)
test_data  = CreateDataset(X_test,y_test)

Trainloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
Testloader = DataLoader(test_data, batch_size=batch_size)


###### Entrenamiento de la red ######
net = lstm_model()
print(net)
net.cuda()
# Funsión de pérdida 
loss_fnc = nn.CrossEntropyLoss()

# Algoritmo de optimización
optimizer = optim.Adam(net.parameters(), lr=0.000005)

#Entrenamiento
Num_epoch = 300

# Log para perdida
record_loss_train = []
record_loss_test = []

for epoch in tqdm(range(Num_epoch)):  # iterar Num_epoch
    net.train()  # Modo de entrenamiento
    loss_train = 0
    for j, (x, t) in enumerate(Trainloader):  # obtener mini-batch de entrenamiento
        x, t = x.cuda(), t.cuda()  # Los datos envian a GPU
        y = net(x)
        loss = loss_fnc(y, t)
        loss_train += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_train /= j+1
    record_loss_train.append(loss_train)

#   Evaluacion
    net.eval()  # Modo de evaluación
    loss_test = 0
    
    for j, (x, t) in enumerate(Testloader):  # obtener mini-batch de prueba
        x, t = x.cuda(), t.cuda()
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
            print("El mejor modelo ha sido actualizado")
            torch.save(net.state_dict(), save_ruta)
            min_loss = loss_test


import matplotlib.pyplot as plt

plt.plot(range(len(record_loss_train)), record_loss_train, label="Train")
plt.plot(range(len(record_loss_test)), record_loss_test, label="Test")
plt.legend()

plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()

#### Probar el funcionamiento con los datos de prueba ######

correct = 0
total = 0
net.load_state_dict(torch.load(save_ruta))  # Cargar el mejor modelo guardado

net.cuda()
net.eval()  # modo de evaluación --- significa que no modificar los pesos de conexión
for i, (x, t) in enumerate(tqdm(Testloader)):
    x, t = x.cuda(), t.cuda()  # Operar en GPU
    y = net(x)
    correct += (y.argmax(1) == t).sum().item()
    total += len(x)

print(" ")    
print("Exactitud:", str(np.round(correct/total*100, 4)) + "%")

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


num_params = get_n_params(net)
print(" number of entrenable parameters = ", num_params)

