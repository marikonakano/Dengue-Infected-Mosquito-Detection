# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 12:37:48 2022

Segmentar audio con traslape de N% (N=50, 25)

(1) Realizar segumentación de sonidos
(2) Se guarda las señales segmentadas en las carpetas correspondiente "infectada" y "sano"
(3) displegar señales.


@author: Mariko Nakano
"""
import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import os
import soundfile as sf

P_traslape = 50   # (%) parametro que debe analizar su selección
L_bloque = 48000   # parametro que debe analizar su selección --- Frecuencia de muestreo
L_bloque_nuevo =int(L_bloque*(100-P_traslape)/100)

path_original = "../Dataset/Original/"
path_infectado = "Infectado/"
path_sano = "Sano/"
path_bloque = "../Dataset/Data_train_test/"

###### Parametros de STFT ######
n_fft = 1024
win_length = n_fft   # predeterminado 
hop_length = 256
window = "hann" # predeterminado
n_mels =32


##### Función para dibujar todas las señales ######
def display_wave(path_base, path, N_column, number = None):
    list_files = os.listdir(path_base + path)
    if number !=None:
        list_files= list_files[:number]
        
    N_datos = len(list_files)
    N_row =N_datos//N_column
    N_rest = N_datos - N_column *N_row
    if N_rest != 0:
        N_row += 1
        
    fig, ax = plt.subplots(N_row, N_column, figsize=(26,10))
    file_num=0

    for i in range(N_row-1):
        for j in range(N_column):
            sound_file = list_files[file_num]
            signal,sr=librosa.load(os.path.join(path_base, path, sound_file), sr=None)
            ax[i,j].plot(signal)
            ax[i,j].axis('off')
            ax[i,j].set_title(sound_file)
            file_num+=1
        
    for j in range(N_rest):
        sound_file = list_files[file_num]
        ax[N_row-1,j].plot(signal)
        ax[N_row-1,j].axis('off')
        ax[N_row-1,j].set_title(sound_file)
        file_num+=1

    plt.show()

display_wave(path_original, path_infectado, N_column=3)
display_wave(path_original, path_sano, N_column=3)


###### Dividir señal en varios segumentos y guardar in archivo de carpeta correspondiente #######
def segment_wave(path_base, path_save, path):
    
    list_file = os.listdir(path_base + path)

    for file in list_file:
        signal, sr = librosa.load(os.path.join(path_base, path, file), sr=None)
    
        ###### Segmentación de datos ######
        L = len(signal)
        Num_segments = int(L/L_bloque_nuevo)-1

        bloque = np.zeros((Num_segments, L_bloque))
        for i in range(Num_segments):
            start = i*L_bloque_nuevo
            end = start + L_bloque
            bloque[i,:] = signal[start:end]
            file_b = os.path.splitext(file)[0] + "b" + str(i) + os.path.splitext(file)[1]
            sf.write(os.path.join(path_save, path,file_b), bloque[i,:], sr, )
        
    #####

#segment_wave(path_original, path_bloque, path_infectado)
#segment_wave(path_original, path_bloque, path_sano)

display_wave(path_bloque, path_infectado, N_column=3, number=10)
display_wave(path_bloque, path_sano,N_colum=3, number=10)
