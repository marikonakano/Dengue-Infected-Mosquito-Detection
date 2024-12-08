# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 09:31:12 2023

Testing different LSTM configurations loading trained weights perfroming load_state_dict(torch.load(ruta_de_archivo)))

(1) Obtain test data to construct Test dataloader
(2) Evaluate the selected model
(3) Apply Monte-Carlo Dropout
(4) Get results for data with 100% certainty

@author: Mariko Nakano
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from torch import optim
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##### Model selection
# ruta_model ='../Model/LSTM_Uni_2H_128.pth'
ruta_model ='../Model/LSTM_Uni_2H_256.pth'
# ruta_model ='../Model/LSTM_Uni_2H_512.pth'

# ruta_model ='../Model/LSTM_Bid_2H_128.pth'
# ruta_model ='../Model/LSTM_Bid_2H_256.pth'
# ruta_model ='../Model/LSTM_Bid_2H_512.pth'

clase =["Infected","Healthy"]

###### models ######

class lstm_uni_128(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=32, hidden_size=128, num_layers = 2, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(128,2)
        
    def forward(self, x):
        
        x, _ = self.lstm(x)
        x = self.linear(x[:,-1,:])
        
        return x

class lstm_uni_256(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=32, hidden_size=256, num_layers = 2, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(256,2)
        
    def forward(self, x):
        
        x, _ = self.lstm(x)
        x = self.linear(x[:,-1,:])
        
        return x    

class lstm_uni_512(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=32, hidden_size=512, num_layers = 2, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(512,2)
        
    def forward(self, x):
        
        x, _ = self.lstm(x)
        x = self.linear(x[:,-1,:])
        
        return x        
    

class lstm_bid_128(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=32, hidden_size=128, num_layers = 2, batch_first=True, dropout=0.2, bidirectional=True)
        self.linear = nn.Linear(256,2)
        
    def forward(self, x):
        
        x, _ = self.lstm(x)
        x = self.linear(x[:,-1,:])
        
        return x

class lstm_bid_256(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=32, hidden_size=256, num_layers = 2, batch_first=True, dropout=0.2, bidirectional=True)
        self.linear = nn.Linear(512,2)
        
    def forward(self, x):
        
        x, _ = self.lstm(x)
        x = self.linear(x[:,-1,:])
        
        return x    

class lstm_bid_512(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=32, hidden_size=512, num_layers = 2, batch_first=True, dropout=0.2, bidirectional=True)
        self.linear = nn.Linear(1024,2)
        
    def forward(self, x):
        
        x, _ = self.lstm(x)
        x = self.linear(x[:,-1,:])
        
        return x        
    

        
###### Select model and get trained weights #####

# net = lstm_uni_128()
net = lstm_uni_256()
# net = lstm_uni_512()
# net = lstm_bid_128()
#net = lstm_bid_256()
# net = lstm_bid_512()

net.load_state_dict(torch.load(ruta_model)) 
  
net.to(device)

##### Evaluate model with test data #####

json_path = '../Dataset/Melspec.json'

with open(json_path, 'r') as f:
  data = json.load(f)


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
    

n_class = 2
train_size = 130*2
test_size = 20*2
batch_size= 16

X_train, X_test, y_train, y_test = train_test_split(data['mspec'], data['labels'], test_size=test_size, random_state=42, stratify=data['labels'])

test_data  = CreateDataset(X_test,y_test)

Testloader = DataLoader(test_data, batch_size=batch_size)

correct = 0
total = 0

net.eval()  # Evaluation mode
for i, (x, t) in enumerate(tqdm(Testloader)):
    x, t = x.to(device), t.to(device)  # Operate in GPU
    y = net(x)
    correct += (y.argmax(1) == t).sum().item()
    total += len(x)

print(" ")    
print("Accuracy:", str(np.round(correct/total*100, 4)) + "%")


### Confusion Matrix #####
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score


Testloader_2 = DataLoader(test_data, batch_size=len(test_data))

for (x_all, t_all) in Testloader_2:
    x_all, t_all = x_all.to(device), t_all.to(device)
    y_all = net(x_all)


y_all_label =y_all.argmax(1)
t_all=t_all.cpu()
y_all =y_all.cpu()
y_all_label= y_all_label.cpu()
cm = confusion_matrix(t_all, y_all_label) 


### Normalization
cm_norm = cm/20

##### Display Confusion matrix ######
import seaborn as sns

plt.figure(figsize=(8,6))
sns.heatmap(cm_norm, annot=True,cmap='Blues')

print(classification_report(t_all, y_all_label))

##### Global AUC ######

from sklearn.preprocessing import LabelBinarizer

label_binarizer = LabelBinarizer()
label_binarizer.fit(range(3))

tt = label_binarizer.transform(t_all)[:,:2]     
y_step = torch.heaviside(y_all, torch.tensor([0.0]))
yy = y_step.detach().numpy()                
auc_score = roc_auc_score(tt,yy)   
    
###### ROC Curve ###
from sklearn.metrics import roc_curve, auc

fpr = {}   # false positive rate
tpr = {}   # true positive rate
roc_auc ={}   # AUC

for i in range(n_class):
    fpr[i], tpr[i], _ = roc_curve(tt[:,i], yy[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
plt.subplots(figsize=(16,10))    
for i in range(n_class):
    plt.plot(fpr[i],tpr[i], label =f'{clase[i]}:  AUC={round(roc_auc[i],3)}')

plt.legend(fontsize="12")
plt.xlabel('False Positive Rate')
plt.ylabel('False Negative Rate')
plt.title(f'ROC Curve  con   Macro AUC = {np.round(auc_score,3)} ')
plt.show()

###############################################################################
########   Uncertainty evaluation

N_test_data = len(test_data)  

Testloader_DO = DataLoader(test_data, batch_size=1)

net.train()   # Montecarlo Dropout Mode without update, but with dropout in inference

Iteracion = 10
umbral = 0.9
correct = 0
incertidumbre=0
datos_incertidumbre = []
record_Ratio = []
y_final_record = []
t_final_record = []

for i, (x, t) in enumerate(tqdm(Testloader_DO)):
    with torch.no_grad():  # without update
        x = x.to(device)  # GPU or CPU
        resultados = []
        for j in range(Iteracion):
            y = net(x)
            resultados.append(y.argmax(1).item())
            
        Ratio = np.mean(resultados)
        y_final = int(np.round(Ratio))
        if Ratio<=0.5:
            certidumbre=1.0-Ratio
        else:
            certidumbre = Ratio
        
        if certidumbre == 1.0:
            t_final = t.cpu().numpy()
            if t_final == y_final:
                correct+=1
            t_final_record.append(t_final)
            y_final_record.append(y_final)
        else:
            incertidumbre+=1
            datos_incertidumbre.append(i)
            record_Ratio.append(certidumbre)
            
       
        
####### Data Analysis ######
        
Num_datos_con_certeza =  N_test_data - incertidumbre       
Accuracy = 100 * (correct/Num_datos_con_certeza)
porcentaje_incertidumbre = 100*(incertidumbre/ N_test_data)

print(" ")
print("Number of data with 100% certainty = ", Num_datos_con_certeza)
print("Number of correct data = ", correct)
print("Accuracy = ", Accuracy)
print("Percentage of uncertainty = ", porcentaje_incertidumbre)

### matriz de confusión ##
cm2 =confusion_matrix(t_final_record, y_final_record)
cm2 = cm2/Num_datos_con_certeza

plt.figure(figsize=(8,6))
sns.heatmap(cm_norm, annot=True,cmap='Blues',xticklabels=["infected","healthy"], yticklabels=["infected","healthy"])
plt.xlabel("predicted", fontsize=14)
plt.ylabel("GT", fontsize = 14)
plt.title("LSTM with 2 layers of 256 neurons")

print(classification_report(t_final_record, y_final_record, digits=4))
