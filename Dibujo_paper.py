# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 16:39:49 2024

Realizar graficas para articulo
@author: Mariko Nakano
"""

import matplotlib.pyplot as plt
import numpy as np

#### grafica -1
X= [1,2,3,4,5]
LSTM_acc = [81.5, 92.5, 90.0, 86.5, 84.0]
BiLSTM_acc = [90, 92.5, 88.5, 83.5, 84.0]

plt.plot(X, LSTM_acc, c="blue", marker ="o", label="LSTM")
plt.plot(X, BiLSTM_acc,c="red", marker ="o", label="Bi-LSTM")
plt.legend(fontsize=16)
plt.xticks([1,2,3,4,5], fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Number of Layers", fontsize=20)
plt.ylabel("Accuracy (%)", fontsize=20)
plt.ylim(75,100)
plt.show()

#### Grafica-2
X= [1,2,3,4,5]
LSTM_num =[0.3, 0.82, 1.35, 1.87, 2.40]
BiLSTM_num = [0.59, 2.17, 3.74, 5.33, 6.90]
plt.plot(X, LSTM_num, c="blue", marker ="o", label="LSTM")
plt.plot(X, BiLSTM_num,c="red", marker ="o", label="Bi-LSTM")
plt.legend(fontsize=16)
plt.xticks([1,2,3,4,5], fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Number of Layers", fontsize=20)
plt.ylabel("Trainable params ($x10^6$)", fontsize=15)
plt.ylim(0, 10)
plt.show()

#### Grafica-3
X= [32,64,128,256,512]
LSTM_acc =[73.4, 85.5, 88.0, 92.5, 93.5]
BiLSTM_acc = [77.5, 84.0, 89.5, 92.5, 95.5]
plt.plot(X, LSTM_acc, c="blue", marker ="o", label="LSTM")
plt.plot(X, BiLSTM_acc,c="red", marker ="o", label="Bi-LSTM")
plt.legend(fontsize=17)
plt.xticks([32,64,128,256,512], fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel("Number of hidden neurons", fontsize=17)
plt.ylabel("Accuracy (%)", fontsize=20)
plt.ylim(70,100)
plt.show()


#### Grafica-4
X= [32,64,128,256,512]
LSTM_num =[0.017, 0.058, 0.22, 0.82, 3.22]
BiLSTM_num = [0.042,0.149,0.561,2.17, 8.54]
plt.plot(X, LSTM_num, c="blue", marker ="o", label="LSTM")
plt.plot(X, BiLSTM_num,c="red", marker ="o", label="Bi-LSTM")
plt.legend(fontsize=17)
plt.xticks([32,64,128,256,512], fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel("Number of hidden neurons", fontsize=17)
plt.ylabel("Trainable params ($x10^6$)", fontsize=15)
plt.ylim(0,10)
plt.show()

#### Grafica-5   Dropout
X = [0.1, 0.2, 0.3, 0.4, 0.5]
LSTM_acc = [84.5, 92.5, 85.0, 84.5, 81.0]
BiLSTM_acc = [91.5,92.5, 92.5, 88.0,91.5]
plt.plot(X, LSTM_acc, c="blue", marker ="o", label="LSTM")
plt.plot(X, BiLSTM_acc,c="red", marker ="o", label="Bi-LSTM")
plt.legend(fontsize=17)
plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5], fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel("Dropout rate", fontsize=17)
plt.ylabel("Accuracy (%)", fontsize=15)
plt.ylim(70, 100)
plt.show()
