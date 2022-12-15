import os
import smtplib
import struct
import time
from email.header import Header
from email.mime.text import MIMEText
from random import random

import numpy as np
import psutil

from model import ResNet as Model
from collections import OrderedDict
# idx1 represents label file
# idx3 represents image file
# download url https://www.westernsydney.edu.au/icns/reproducible_research/publication_support_materials/emnist

class DecodeUbyte():
    def __init__(self,idx1_file_name,idx3_file_name,mapping_name):
        self.buf_img = open(idx3_file_name,"rb").read()
        self.buf_label = open(idx1_file_name,"rb").read()
        self.dict = self.parse_mapping(mapping_name)
        self.num = self.get_num()
        
    
    def get_num(self):
        _,image_num = struct.unpack_from(">II",self.buf_img,0)
        _,label_num = struct.unpack_from(">II",self.buf_label,0)
        assert image_num==label_num
        return image_num-1
    
    def get_image(self,id):
        image_index = struct.calcsize(">IIII")+struct.calcsize(">784B")*id
        img = struct.unpack_from(">784B",self.buf_img,image_index)
        img = np.reshape(img,(28,28))
        img = np.rot90(img,-1)
        return img
    
    def get_label(self,id):
        label_index = struct.calcsize(">II")+struct.calcsize("B")*id
        label = struct.unpack_from("B",self.buf_label,label_index)
        return label
    
    def parse_mapping(self, mapping_name):
        dict_arr = np.loadtxt(mapping_name)
        dict = {}
        for i in dict_arr:
            dict[i[0]] = int(i[1])
        return dict     
    
    def get_mapping(self,key):
          return self.dict[key]

import torch


def compute_accuracy(possibility, label):
    sample_num = label.size(0)
    _,index = torch.max(possibility,1)
    correct_num = torch.sum(label==index)
    return (correct_num/sample_num).item()
    

import torch.nn as nn
from typing import List

Model_Path_List = List[str]
def model_aggregate(models_paths:Model_Path_List,server_model_path:str)->nn.Module:
    result_model = Model(num_class=10)
    dictKeys = result_model.state_dict().keys()
    
    server_model = torch.load(server_model_path).cpu()
    
    state_dict = OrderedDict()
    model_list = []
    for model_path in models_paths:
        model_list.append(torch.load(model_path))
    
    
    for key in dictKeys:
        id = 0
        for model in model_list:
            if id==0:
                state_dict[key] = model.state_dict()[key]
            if id!=0:
                state_dict[key] = state_dict[key] + model.state_dict()[key]
            id+=1
        
        state_dict[key] = state_dict[key]/len(models_paths)
    
    # for key in dictKeys:
        # state_dict[key] = server_model.state_dict()[key]+state_dict[key]
    
    result_model.load_state_dict(state_dict)
    return result_model

def send_email(receiver, message):
    sender = 'wuwang365@qq.com'
    receivers = [receiver]
    message['From'] = Header("wuwang365", "utf-8")
    message['To'] = Header("wuwang365", "utf-8")
    message["Subject"] = Header("自动邮件提醒", "utf-8")
    try:
        smt = smtplib.SMTP()
        smtp = smtplib.SMTP_SSL("smtp.qq.com")
        smtp.ehlo("smtp.qq.com")
        smtp.login("wuwang365@qq.com", "iitrbcjkgoonihii")
        smtp.sendmail(sender, receivers, message.as_string())
        smtp.quit()
    except smtplib.SMTPException:
        print("faild!")
        smtp.quit()

def check_memory():
    with open("memory_log","w") as f:
        f.close()
    while(1):
        memory = psutil.Process(os.getpid()).memory_info().rss/1024/1024/1024
        with open("memory_log","a") as f:
            f.write("{:.2f},".format(memory))
        time.sleep(1)
        
        
import json


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

import glob
import pickle


def process_data():
    files = glob.glob("./data/data_*")
    result = {}
    for i in range(0,10):
        result[i] = []
        
    for file in files:
        dict = unpickle(file)
        for index,i in enumerate(dict[b"labels"]):
            result[i].append(dict[b"data"][index])
    
    del_prob = 0.98
    for i in range(0,5):
        label_index_1 = 2*i
        label_index_2 = 2*i+1
        
        result = test_dataset_build(label_index_1,label_index_2,result)
        
        for id in range(49):
            save_name = "class_{:s}_class_{:s}_id_{:s}".format(str(label_index_1),str(label_index_2),str(id))
            client_info = {}
            
            client_sample_id = 0
            for item in result[label_index_1]:
                if np.random.rand()>del_prob:
                    info_cell = {"label":label_index_1,"data":item}
                    client_info[client_sample_id] = info_cell
                    client_sample_id+=1
            
            for item in result[label_index_2]:
                if np.random.rand()>del_prob:
                    info_cell = {"label":label_index_2,"data":item}
                    client_info[client_sample_id] = info_cell
                    client_sample_id+=1
            
            with open("client_data/{:s}".format(save_name),"wb") as f:
                
                pickle.dump(client_info,f)
                
def test_dataset_build(index_1,index_2,result_data):
    client_info = {}
    client_sample_id = 0
    save_name = "test_data_class_{:d}_class_{:d}".format(index_1,index_2)
    for i in (index_1,index_2):
        samples_index = np.random.choice(len(result_data[i]),size=(1000),replace=False)
        for index in samples_index:
            info_cell = {"label":i,"data":result_data[i][index]}
            client_info[client_sample_id] = info_cell
            client_sample_id+=1
        
        result_not_select = []
        result_all = result_data[i]
        for (index,item) in enumerate(result_all):
            if index not in samples_index:
                result_not_select.append(item)
        result_data[i] = result_all
        
    with open("test_data/{:s}".format(save_name),"wb") as f:
        pickle.dump(client_info,f)
        
    return result_data

        
    
        

def process_global_data():
    files = glob.glob("./data/data_*")
    result = {}
    for i in range(0,10):
        result[i] = []
        
    for file in files:
        dict = unpickle(file)
        for index,i in enumerate(dict[b"labels"]):
            result[i].append(dict[b"data"][index])
    del_prob = 0.98
    client_info = {}
    id = 0
    for i in range(10):
        
        for item in result[i]:
            if np.random.rand()>del_prob:
                info_cell = {"label":i,"data":item}
                client_info[id] = info_cell
                id+=1
    
    with open("global_data/global_data.json","wb") as f:
        pickle.dump(client_info,f)
        


def global_init():
    model = Model(num_class=10)
    init_weight_path = "server_weight/0.pth"
    torch.save(model,init_weight_path)
    return init_weight_path
            
            
            
                
def re_sort(array:list):
    import re
    array.sort(key=lambda x:int("".join(re.findall("\d",x))))
    return array

def check_grad(model:nn.Module):
    for param in model.parameters():
        print(param.grad)
        
def fusion_model(server_model:nn.Module,loacl_model:nn.Module):
    model = Model(num_class=10)
    
    state_dict = OrderedDict()
    
    server_state_dict = server_model.state_dict()
    local_state_dict = loacl_model.state_dict()
    
    for key in model.state_dict().keys():

        if "fc" in key:
            state_dict[key] = local_state_dict[key]
        else:
            state_dict[key] = server_state_dict[key]
    model.load_state_dict(state_dict)
    
    return model

def model_aggregate_max(models_paths:Model_Path_List)->nn.Module:
    result_model = Model(num_class=10)
    dictKeys = result_model.state_dict().keys()
    
    state_dict = OrderedDict()
    model_list = []
    for model_path in models_paths:
        model_list.append(torch.load(model_path).cpu())
    
    
    for key in dictKeys:
        state_dict[key] = torch.zeros(size=result_model.state_dict()[key].shape)
        for model in model_list:
            cur_model_weight = model.state_dict()[key]
            cur_abs_model_weight = torch.abs(cur_model_weight)
            compare_flag = cur_abs_model_weight>state_dict[key]
            state_dict[key] = compare_flag*cur_model_weight+(torch.logical_not(compare_flag))*state_dict[key]
        
    result_model.load_state_dict(state_dict)
    return result_model

def model_aggregate_maxdiff(models_paths:Model_Path_List,server_model_path:str)->nn.Module:
    server_model = torch.load(server_model_path).cpu()
    server_dict = server_model.state_dict()
    
    result_model = Model(num_class=10)
    
    dictKeys = result_model.state_dict().keys()
    
    state_dict = OrderedDict()
    model_list = []
    for model_path in models_paths:
        model_list.append(torch.load(model_path).cpu())
    
    
    for key in dictKeys:
        if "weight" in key or "bias" in key:
            state_dict[key] = torch.zeros(size=result_model.state_dict()[key].shape)
            diff = torch.zeros(size=result_model.state_dict()[key].shape)
            
            for model in model_list:
                cur_model_weight =  model.state_dict()[key]
                
                # 计算当前模型与服务器模型之间的差异            
                cur_diff = torch.abs(server_dict[key]-cur_model_weight)
                
                weight_regular = cur_diff/torch.mean(cur_diff)
                weight_regular.clamp_(min=1)                
                # 更新差异矩阵
                # compare_flag = diff>cur_diff
                # diff = compare_flag*diff+(torch.logical_not(compare_flag))*cur_diff
                
                # 根据差异大小情况，更新权值
                # state_dict[key] = compare_flag*state_dict[key]+(torch.logical_not(compare_flag))*cur_model_weight
                state_dict[key] = state_dict[key]+weight_regular*cur_model_weight
            
            state_dict[key] = state_dict[key]/len(model_list)
        else:
            state_dict[key] = server_dict[key]
        
    result_model.load_state_dict(state_dict)
    return result_model

if __name__=="__main__":       
    process_data()