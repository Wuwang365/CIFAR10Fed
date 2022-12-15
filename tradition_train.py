import torch
from data import EMNIST_Dataset_filter
from torch.utils.data import DataLoader
import torch.nn as nn
from loss import model_reg_loss,paired_model_reg_loss,model_regular_rev
import os
import numpy as np
from util import model_aggregate, global_init,check_grad
from test import test,test_server_model
import re


def tradition_train(info_path:str,global_weight_path,local_epoch=50):

    
    batch_size = 64
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=False).cuda()
    
    train_data = EMNIST_Dataset_filter(info_path)
    
    train_data_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    cross_entropy = nn.CrossEntropyLoss()
    acc,entropy =0,0
    
    for epoch in range(local_epoch):
        for batch_num,(image,label) in enumerate(train_data_loader):
                
                image = image.cuda()
                label = label.cuda()
                output = model(image)
                
                loss_entropy = cross_entropy(output,label)
                # print(loss)
                
                optimizer.zero_grad()
                loss_entropy.backward()
                optimizer.step()
                
        print(loss_entropy)
        

if __name__ == "__main__":
    tradition_train(r"global_data\global_data.json",r"server_weight\0.pth")
        

