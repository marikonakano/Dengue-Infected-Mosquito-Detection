# Dengue-Infected Mosquito Detection with Uncertainty Evaluation of the Wingbeats using Monte Carlo Dropout

## Graphical Abstract

<img width="1300" alt="Graphical_Abstract_HW" src="https://github.com/user-attachments/assets/66647b72-87b0-490b-851b-88d90ba7eb5a">

## File Description

### Training_LSTM.py  
Trainig LSTM using Melspec data of wingbeat signal.
Select different configurations of LSTM and Bi-LSTM.

### Testing Testing_LSTM_MC_DO.py    
Testing LSTM and Monte-Carlo Dropout and uncertainty evaluation.
Select a pre-trained model of different configurations of LSTM and Bi-LSTM.

## Usage
The code uses Pytorch. Please install the Pytorch Package.

Unzip Melspec.zip in the Dataset Folder and locate it in the Dataset folder.

Select pre-trained data in the Model Folder if only the pre-trained LSTM is proven.

## Principal Results
After uncertainty evaluation, we obtained the following confusion matrix.

[Bi-LSTM] <img width="202" alt="Confusion Matrix Bi-LSTM" src="https://github.com/NakanoMariko/Test/blob/main/Confusion%20Matrix%20Bi-LSTM.png">     [LSTM]  <img width="202" alt="Confusion Matrix LSTM" src="https://github.com/NakanoMariko/Test/blob/main/Confusion%20Matrix%20LSTM.png">

## Citation


