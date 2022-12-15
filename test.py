import numpy as np
from data import EMNIST_Dataset_filter
from torch.utils.data import DataLoader
import torch
from util import compute_accuracy,re_sort
import torch.nn.functional as F
import glob
import torch.nn as nn
import torchvision

def test(model:nn.Module,data_path:str):
    
    data_path = data_path.replace("client_data/","test_data/test_data_")
    suffix = data_path.split("_")[-1]
    data_path = data_path.replace("_id_"+suffix,"")
    
    model.eval()
    
    train_data = EMNIST_Dataset_filter(data_path)
    train_data_loader = DataLoader(train_data,batch_size=64,shuffle=True)
    cross_entropy = nn.CrossEntropyLoss()
    entropy = 0
    acc = 0
    
    for batch_num,(image,label) in enumerate(train_data_loader):
            image = image.cuda()
            label = label.cuda()
            output = model(image)
            entropy += cross_entropy(output,label).item()
            acc += compute_accuracy(output,label)
    
    entropy/=(batch_num+1)
    acc/=(batch_num+1)
    return acc,entropy
    
def test_server_model(model:nn.Module):
    
    data_path_root = "test_data/*"
    
    model.cuda().eval()
    
    cross_entropy = nn.CrossEntropyLoss()
    
    entropy = 0
    acc = 0
    
    for data_file_path in glob.glob(data_path_root):
        train_data = EMNIST_Dataset_filter(data_file_path)
        train_data_loader = DataLoader(train_data,batch_size=64,shuffle=True)
        
        entropy_per_data = 0
        acc_per_data = 0
        
        for batch_num,(image,label) in enumerate(train_data_loader):
            image = image.cuda()
            label = label.cuda()
            output = model(image)
            entropy_per_data += cross_entropy(output,label).item()
            acc_per_data += compute_accuracy(output,label)
        
        entropy_per_data/=(batch_num+1)
        acc_per_data/=(batch_num+1)
        
        print("accuracy: {:.3f}\nentropy: {:.3f}"
              .format(acc_per_data,entropy_per_data))
        
        entropy+=entropy_per_data
        acc+=acc_per_data
        
     
    entropy/=len(glob.glob(data_path_root))
    acc/=len(glob.glob(data_path_root))

    print("average accuracy: {:.3f}\naverage entropy: {:.3f}".format(acc,entropy))
    
    return acc,entropy
        

import json

def test_all_server_weight():
    files = glob.glob("server_weight/*")
    files = re_sort(files)[0:2300]
    training_info = {}
    
    for file in files:
        weight_key = file.replace("server_weight\\","")
        
        model = torch.load(file)
        acc, entropy = test_server_model(model)
        torch.cuda.empty_cache()
        
        print(weight_key)
        training_info[weight_key] = {
            "acc":acc,
            "entropy":entropy
        }
    
    with open("./training_info.json","w") as f:
        f.write(json.dumps(training_info,indent=4))
    
    


if __name__=="__main__":
    test_all_server_weight()
    
    