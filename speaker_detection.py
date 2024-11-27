import time
import os
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display
 
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.utils
from dataset.audio_dataset import AudioDatasets
from model.model import Net1D
from utils.utils import train_fn,test_fn

speaker_label = {'Jens_Stoltenberg' : 0,
                 'Benjamin_Netanyau': 1,                 
                 'Julia_Gillard' : 2,
                 'Magaret_Tarcher' : 3,
                 'Nelson_Mandela' : 4,
                }
sr = 16000

def read_file_label(path_name,speaker_label):
    file_label_list = []
    file_list = []
    label_list = []

    for label in speaker_label.keys():
        n_file = len(glob.glob(os.path.join(path_name,list(speaker_label.keys())[0])+"/*.wav"))
        
        for i in range(n_file):
            file_label_list.append((os.path.join(path_name,label,f"{i}.wav"),speaker_label[label]))
        
            file_list.append(os.path.join(path_name,label,f"{i}.wav"))

            label_list.append(speaker_label[label])

    return file_label_list,file_list,label_list

file_label_list,file_list,label_list = read_file_label("16000_pcm_speeches",speaker_label=speaker_label)

X_train,X_valid,y_train,y_valid = train_test_split(file_list,label_list,test_size=0.2,random_state=31)

train = [(X_train[idx],y_train[idx]) for idx in range(len(X_train))]
valid = [(X_valid[idx],y_valid[idx]) for idx in range(len(X_valid))]

batch_size = 32

train_loader = torch.utils.data.DataLoader(AudioDatasets(train,sr=sr),batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(AudioDatasets(valid,sr=sr),batch_size=batch_size)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Net1D().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(),lr=1e-3)

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\\n-------------------------------")
    train_fn(dataloader=train_loader,model=model,loss_fn=loss_fn,optimizer=optimizer,device=device)
    test_fn(dataloader=test_loader,model=model,loss_fn=loss_fn,device=device)
print("Done!")








