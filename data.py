from typing import Dict
from torch.utils.data import Dataset
from util import DecodeUbyte
import torch
import numpy as np
import json
from PIL import Image
import pickle

      
      
class EMNIST_Dataset_filter(Dataset):
    def __init__(self,data_path,del_per=0) -> None:
        super().__init__()
        self.del_per = del_per
        self.all_data = self.read_data(data_path = data_path)
        
            
            
    def read_data(self, data_path):
        data = unpickle(data_path)
        return data
    
    def __getitem__(self, key):
        data = self.all_data[key]
        label = data["label"]
        image = data["data"]
        image = torch.from_numpy(np.asarray(image)).reshape(3,32,32)/255.0
        return image,label
    
    def __len__(self)->int:
        return len(self.all_data.keys())


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
