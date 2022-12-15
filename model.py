from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
import torch

class ResBlock(nn.Module):
    def __init__(self,inchannels,outchannels,stride=1):
        super().__init__()
        self.convBlock = nn.Sequential(
            nn.Conv2d(inchannels,outchannels,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannels,outchannels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(outchannels)
        )
        self.shortcut = nn.Sequential()
        if inchannels!=outchannels or stride!=1:
            self.shortcut.add_module("shortcut",nn.Conv2d(inchannels,outchannels,kernel_size=1,stride=stride))
        
    def forward(self,x):
        out = self.convBlock(x)
        out += self.shortcut(x)
        return F.relu(out)
    
class ResNet(nn.Module):
    def __init__(self,num_class=47):
        super().__init__()
        self.pool = nn.MaxPool2d(stride=2,kernel_size=3,padding=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.layer1 = ResBlock(64,64,1)
        self.layer2 = nn.Sequential(ResBlock(64,128,2),ResBlock(128,128,1))
        self.layer3 = nn.Sequential(ResBlock(128,256,2),ResBlock(256,256,1))
        self.layer4 = nn.Sequential(ResBlock(256,512,2),ResBlock(512,512,1))
        self.fc = nn.Linear(512,num_class)
        
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = F.avg_pool2d(x5,2)
        x7 = x6.view(x6.size(0),-1)
        x8 = self.fc(x7)
        # output = F.softmax(x8, dim=1)
        return x8
        
class SimpleNet(nn.Module):
    def __init__(self,num_class) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.pool = nn.AvgPool2d(stride=2,kernel_size=3,padding=1)
        self.fc = nn.Linear(512,num_class)
        
        
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.pool(x1)
        x3 = self.conv2(x2)
        x4 = self.pool(x3)
        x5 = self.pool(x4)
        x6 = x5.view(x5.size(0),-1)
        x7 = self.fc(x6)
        F.softmax(x7, dim=1)
        return x7
        
        
# 建立一个四层感知机网络
class MLP(torch.nn.Module):  
    def __init__(self,num_class):
        super(MLP,self).__init__()    # 
        # 初始化三层神经网络 两个全连接的隐藏层，一个输出层
        self.fc1 = torch.nn.Linear(3072,512)  # 第一个隐含层  
        self.fc2 = torch.nn.Linear(512,128)  # 第二个隐含层
        self.fc3 = torch.nn.Linear(128,num_class)   # 输出层
        
    def forward(self,din):
        din = din.view(-1,32*32*3)       # 将一个多行的Tensor,拼接成一行
        dout = F.relu(self.fc1(din))   # 使用 relu 激活函数
        dout = F.relu(self.fc2(dout))
        dout = F.softmax(self.fc3(dout), dim=1)  # 输出层使用 softmax 激活函数
        # 10个数字实际上是10个类别，输出是概率分布，最后选取概率最大的作为预测值输出
        return dout

        
        