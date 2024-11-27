import torch
import librosa
import numpy as np


class AudioDatasets(torch.utils.data.Dataset):
    
    def __init__(self,file_label_list, sr):
        self.path = file_label_list
        self.sr = sr
    
    def __len__(self):
        return len(self.path)
    
    def __getitem__(self,idx):
        data,_ = librosa.load(self.path[idx][0],sr=self.sr)
        data = data.reshape(1,self.sr)
        label = np.array(self.path[idx][1])
        return data, label