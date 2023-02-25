import torch
import torch.nn as nn
import numpy as np
import random
from common import function
Datasoftmax = nn.Softmax(dim = 1)
Datasigmoid = nn.Sigmoid()
Datarelu = nn.ReLU()
datatanh = nn.Tanh()
#初始化LSTM模型
inputsize  = int(function.getConfig("Config",'model','inputsize'))
hiddensize  = int(function.getConfig("Config",'model','hiddensize'))
linearsize = int(function.getConfig("Config",'model','linearsize'))
numlayers = int(function.getConfig("Config",'model','numlayers'))
class lstm(nn.Module):
    def __init__(self,input_size = inputsize,output_size = 1):#输入参数量和输出参数量
        super().__init__()
        self.hidden_layer_size = hiddensize #lstm隐藏层大小
        self.linearsize = linearsize #最后全连接层的大小
        self.attentionLinear = nn.Linear(input_size, input_size)#特征注意力层
        self.lstm1 = nn.LSTM(input_size,self.hidden_layer_size,num_layers=numlayers,bidirectional=False)#LSTM层
        #最后的全连接层
        self.reg = nn.Sequential(
            nn.Linear(self.hidden_layer_size, self.linearsize),
            nn.Tanh(),
            nn.Linear(self.linearsize,output_size)
        )
        #初始化单元细胞
        if torch.cuda.is_available():
            self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size).cuda(),torch.zeros(1,1,self.hidden_layer_size).cuda())
        else:
            self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size), torch.zeros(1, 1, self.hidden_layer_size))
    #前馈函数，每次读取四个时间序列的值，最后输出最后一个时间点的预测值
    def forward(self,input_seq):
        #特征注意力层
        attenetionLinearOut = self.attentionLinear(input_seq)
        attenetionLinearOut = Datasigmoid(attenetionLinearOut)
        attenetionLinearOut = Datasoftmax(attenetionLinearOut)
        newInput = input_seq *attenetionLinearOut
        #初始化单元细胞
        if torch.cuda.is_available():
            self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size).cuda(), torch.zeros(1, 1, self.hidden_layer_size).cuda())
        else:
            self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size), torch.zeros(1, 1, self.hidden_layer_size))
        #LSTM的输出（因为加了4个时间序列，所以有4个）
        lstm_out1,self.hidden_cell =self.lstm1(newInput.view(len(newInput),1,-1),self.hidden_cell)
        #LSTM输出加入到全连接（输出4个，只取最后一个，因为最后一个包含了前三个的信息）
        predictions = self.reg(lstm_out1.view(len(lstm_out1),-1))
        return predictions[3]
#设定时间种子函数
def setup_seed(seed):
    torch.manual_seed( seed )
    torch.cuda.manual_seed_all( seed )
    np.random.seed( seed )
    random.seed( seed )
#初始化模型函数
def getModel():
    model = lstm()
    if torch.cuda.is_available():
        model = model.cuda()
    return model
#初始化损失函数
def getLoss_function():
    loss_function = nn.MSELoss()
    if torch.cuda.is_available():
        loss_function = loss_function.cuda()
    return loss_function
#初始化优化器函数
def getOptimizer(model,LR):
    optimizer = torch.optim.Adam(model.parameters(),lr=LR)
    return optimizer