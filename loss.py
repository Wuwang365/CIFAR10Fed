import torch
from torch import nn as nn
from model import ResNet,SimpleNet

def model_aggregate(model_1:nn.Module,model_2:nn.Module)->nn.Module:
    result_model = SimpleNet(numClass=10)
    state_dict = result_model.state_dict()
    for key in state_dict.keys():
        print(key)
        state_dict[key] = (model_1.state_dict()[key]+model_2.state_dict()[key])/2
    result_model.load_state_dict(state_dict)
    return result_model

def model_reg_loss(model:nn.Module):
    state_dict = model.state_dict()
    weight_loss = 0
    for key in state_dict.keys():
        if len(state_dict[key].shape)==4:
            weight_sum = torch.sum(torch.abs(torch.mean(state_dict[key],dim=[2,3])))
        if len(state_dict[key].shape)==1:
            weight_sum = torch.sum(torch.abs(state_dict[key]))
        weight_loss+=weight_sum
    return weight_loss

def paired_model_reg_loss(model:nn.Module,demo_model:nn.Module):
    state_dict = model.state_dict()
    demo_sttate_dict = demo_model.state_dict()
    
    weight_loss = 0
    for key in state_dict.keys():
        filter = demo_sttate_dict[key]==0
        weight_filter_result = (state_dict[key]*filter).float()
        weight_loss += torch.mean(torch.abs(weight_filter_result))
    
    return weight_loss



def model_show():
    model = torch.load("cpt/cpt_num_0_4/demo.pth")
    state_dict = model.state_dict()
    for key in state_dict.keys():
        print(key)
        
def model_purn():
    model:nn.Module = torch.load("cpt/cpt_num_0_4/demo.pth")
    state_dict = model.state_dict()
    zero_num = 0
    for key in state_dict.keys():
        if len(state_dict[key].shape)==4:
            state_dict[key] = purn_filter(state_dict[key])
        if len(state_dict[key].shape)==1:
            state_dict[key] = purn_weight(state_dict[key])
        
    model.load_state_dict(state_dict)
    torch.save(model,"./cpt/pure_result/result.pth")
    
def purn_filter(layer_dict):
    
    layer_dict_1_mean = torch.abs(torch.mean(layer_dict,dim=[1,2,3]))
    # for index,i in enumerate(layer_dict_1_mean):
    #     if i<1e-4:
    #         layer_dict[index] = torch.zeros(layer_dict[index].shape)
    num = 0
    purn_num = 0
    layer_dict_2_mean = torch.abs(torch.mean(layer_dict,dim=[2,3]))
    for index_1,out_channel in enumerate(layer_dict_2_mean):
        for index_2,i in enumerate(out_channel):
            num+=1
            if i<5e-4:
                purn_num+=1
                layer_dict[index_1][index_2] = torch.zeros(layer_dict[index_1][index_2].shape)
    print(num)
    print(purn_num)    
    return layer_dict

def purn_weight(layer_dict):
    for index,i in enumerate(layer_dict):
        if torch.abs(i)<1e-3:
            layer_dict[index] = torch.tensor(0,dtype=float)
    return layer_dict
        

def pruning():
    model:nn.Module = torch.load("cpt/prun_result/result.pth")
    state_dict = model.state_dict()
    for key in state_dict.keys():
        param = state_dict[key]
        param[param!=0] = 1
        state_dict[key] = param
    model.load_state_dict(state_dict)
    torch.save(model,"./cpt/prun_result/result_binary.pth")
    
def model_regular_rev(model:nn.Module):
    state_dict = model.state_dict()
    weight_loss = 0
    for key in state_dict.keys():
        
        min_value = torch.min(state_dict[key])
        max_value = torch.max(state_dict[key])
        
        if "weight" in key:
            weight = state_dict[key]
            
            weight_loss+=torch.sum(torch.abs(weight))
        
        # if len(state_dict[key].shape)==1:
        #     element_num = weight_mean.numel()
        #     step = 1/element_num
        #     weight_hand = (torch.arange(0,1,step).cuda())*(max_value-min_value)+min_value
            
        #     weight_weight = torch.reshape(weight_hand, shape=weight_mean.shape)
        #     weight_mean_balance = weight_mean - weight_weight
            
        #     weight_loss+=torch.sum(torch.abs(weight_mean_balance))
        
    return weight_loss
            
            

        
if __name__=="__main__":
    pruning()