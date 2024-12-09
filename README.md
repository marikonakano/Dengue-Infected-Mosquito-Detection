# Dengue-Infected Mosquito Detection with Uncertainty Evaluation of the Wingbeats using Monte Carlo Dropout

## Graphical Abstract

<img width="1300" alt="Graphical_Abstract_HW" src="https://github.com/user-attachments/assets/66647b72-87b0-490b-851b-88d90ba7eb5a">

## File Description

### Training_LSTM.py  
Train LSTM using Melspec data of the mosquito's wingbeat signal.
Select different configurations of LSTM and Bi-LSTM for training.

### Testing Testing_LSTM_MC_DO.py    
Test LSTM and carry out the uncertainty evaluation based on the Monte-Carlo Dropout.
Select a pre-trained model of different configurations of LSTM and Bi-LSTM.

## Usage
The code uses Pytorch. Please install the Pytorch Package.

If one of the different configurations of LSTM, please Unzip Melspec.zip in the Dataset Folder and locate it in the local Dataset folder.

If you want to prove the different configurations of pre-trained LST, please select pre-trained weight data in the Model Folder.

## Principal Results
After uncertainty evaluation, we obtained the following confusion matrix.

[Bi-LSTM] <img width="202" alt="Confusion Matrix Bi-LSTM" src="https://github.com/NakanoMariko/Test/blob/main/Confusion%20Matrix%20Bi-LSTM.png">     [LSTM]  <img width="202" alt="Confusion Matrix LSTM" src="https://github.com/NakanoMariko/Test/blob/main/Confusion%20Matrix%20LSTM.png">

## Dataset
The dataset is being expanded to include mosquito wingbeat signals under different conditions, such as mosquito age, feeding condition, different dengue virus serotypes (type 2 and type 3), and environmental conditions(temperature and humidity).

The link to the dataset will be available in this repository.

### Example of the Wingbeat signals
The wingbeat signal of a healthy Ae. aegypti mosquito and that of a Dengue 2-infected Ae. aegypti mosquito are available in the "Healthy and Infected Mosquitoes" folder.

## Citation


