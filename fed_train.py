import torch
from data import EMNIST_Dataset_filter
from torch.utils.data import DataLoader
import torch.nn as nn
from loss import model_reg_loss,paired_model_reg_loss,model_regular_rev
import os
import numpy as np
from util import model_aggregate, global_init,check_grad, model_aggregate_max, model_aggregate_maxdiff
from test import test,test_server_model
import re
from util import fusion_model


def client_train(info_path:str,global_weight_path,learning_rate,local_epoch=5):
    
    client_name = info_path.replace("\\","/").split("/")[-1]
    
    # print("Client: {:s} is training".format(client_name))
    
    batch_size = 64
    server_model = torch.load(global_weight_path).cuda()
        
    model = server_model
    save_path = "./client_weight/{:s}.pth".format(client_name)
    
    train_data = EMNIST_Dataset_filter(info_path)
    
    train_data_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
    
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
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
    
        group_name = re.findall("class_\d_class_\d",client_name)[0]
        with open("log_dir/"+group_name+".txt","a+") as f:
            f.write("entropy:{:.3f}\n".format(loss_entropy.item()))
        
        
    
    acc, entropy = test(model,info_path.replace("\\","/"))
    with open("log_dir/"+group_name+".txt","a+") as f:
        f.write("acc:{:.3f}\n".format(acc))
    torch.save(model.cpu(),save_path)
    return save_path, acc, entropy

import glob
def server_train():
    
    
    
    client_infos = glob.glob("client_data/*")
    client_set = []
    
    global_weight_path = global_init()
    
    for client_info in client_infos:
        client_set.append(client_info)
    
    epoch = 0
    
    while(epoch<3000):
        epoch+=1
        print("Server epoch: {:d}...".format(epoch))
        training_set = np.random.choice(client_set,10)
        save_paths = []
        accs = []
        entropys = []
        import time
        
        for training_client in training_set:
            epoch_num = np.random.randint(low=1,high=5,size=1)[0]
            
            save_path, acc, entropy = client_train(training_client,global_weight_path,1e-3,1)
            
            save_paths.append(save_path)
            accs.append(acc)
            entropys.append(entropy)
            
        server_model = model_aggregate(save_paths,global_weight_path)
        global_weight_path = "server_weight/{:s}.pth".format(str(epoch))
        
        test_server_model(server_model)
        
        # print("acc is {:.3f}".format(np.mean(accs)))
        # print("entropy is {:.3f}".format(np.mean(entropys)))
        
        torch.save(server_model,global_weight_path)


if __name__=="__main__":
    
    server_train()
        
            
        
