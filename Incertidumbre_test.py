# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:28:43 2024

A partir del modelo de red guardada, detecta imagenes de entradas con mayor incertidumbre

@author: Mariko Nakano
"""
import torch
import torch.nn as nn

# Definir device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import collections

# Transform --> Solamente convertir a Tensor
transform = transforms.Compose([
    transforms.ToTensor()                     # convertir en torch.tensor
])

# Obtener datos de prueba MNIST
test_dataset_original = datasets.MNIST(
    './data',                                 # directorio para guardar datos
    train = False,                            # test data
    transform = transform                     # Convertir en torch
)

# Generar test_loader con batch size =1
test_loader_DO = DataLoader(
    test_dataset_original,
    batch_size = 1,
    shuffle=False  
    )

### definir una red sensilla ###
class Red_DO(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        z = self.encoder(x)
        return z

net = Red_DO().to(device) 
ruta = "./Modelos/Red_MNIST.pth"
net.load_state_dict(torch.load(ruta)) 

#### Evaluar incertidumbre ####

net.train()   # en lugar de modo de evaluación, poner modo de prueba

y_final=[]
count=0
for j, (x, t) in enumerate(test_loader_DO): 
    with torch.no_grad():
        x, t = x.to(device), t.to(device)
        resultados = []
        for i in range(10):
            y = net(x)
            resultados.append(y.argmax(1).item())
        freq = collections.Counter(resultados)  
        if len(freq) == 1:
            y_final.append(resultados[0])
            # if count < 200:
                # plt.imshow(x.cpu().numpy().squeeze(), cmap="gray")
                # plt.axis("off")
                # plt.title(str(resultados[0]))
                # plt.show()
            count+=1
        elif freq[list(freq)[0]] > 5:
            y_final.append(list(freq)[0])
        else:
            y_final.append(-1)
            plt.imshow(x.cpu().numpy().squeeze(), cmap="gray")
            plt.axis("off")
            plt.title(str(list(freq)[:]))
            plt.show()
        
            
#### Analisis ####

#### Numero de entradas con incertidumbre ####
print(f'numero de entrada con incertidumbre: {y_final.count(-1)}')
