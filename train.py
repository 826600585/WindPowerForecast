import random
import numpy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import modelJ00154 as model_pkg
from dataReader import trainDataset
from dataReader import testDataset
import joblib
from common import function

import dataReader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
#---------------------------------构建可训练的数据-----------------------------------#
learnrate  = float(function.getConfig("Config",'model','learnrate'))
backpropagation = int(function.getConfig("Config",'model','backpropagation'))
windnumber  = function.getConfig("Config",'trainfile','windnumber')
maxpower = float(function.getConfig("Config","model","maxpower"))
def train(windfarm_Data,epochs,modelFileOut,verifyData):#训练函数
    trainseqlist = list()#训练集结果列表
    trainlablelist = list()#训练集标签列表
    model= model_pkg.getModel()#初始化模型函数
    loss_function = model_pkg.getLoss_function()#初始化损失函数
    LR=learnrate#初始化学习率
    optimizer = model_pkg.getOptimizer(model,LR)#初始化优化器
    finalacc = 0 # 初始化验证集准确率
    finalVerifyLoss = 9999 #初始化验证集损失值
    Alldataset=trainDataset()
    traindataset=torch.utils.data.Subset(Alldataset,(0,Alldataset.len))
    verifydataset=torch.utils.data.Subset(Alldataset,(Alldataset.len-1000,Alldataset.len))
    train_loader = DataLoader(dataset=traindataset, batch_size=32, shuffle=True)
    verify_loader = DataLoader(dataset=verifydataset, batch_size=32, shuffle=True)
    for i in range(epochs):#开始训练
        count=0 #标签，用于记录固定轮次进行一次反向传播
        loss = 0 #初始化损失值
        totalloss=0 #记录整个训练集损失值
        for k, data in enumerate(train_loader):
            seq,lables= data
            if torch.cuda.is_available():
                seq = seq.cuda()
                lables = lables.cuda()
            y_pred=model(seq)#将特征值输入模型，得到预测值
            y_pred = y_pred.reshape(1, -1) #调整一下形状，使之和标签一样，用于损失值计算
            single_loss = loss_function(y_pred, lables)#计算单步的损失值
            if torch.cuda.is_available():
                y_pred = y_pred.cpu()
                lables = lables.cpu()
            loss = loss+single_loss#将单步损失值相加
            if i == epochs -1:#如果是最后一个轮次，就把最后一次训练的预测结果和标签写进列表，然后可视化，用于查看训练集拟合程度
                trainseqlist.append(y_pred.tolist())
                trainlablelist.append(lables.tolist())
            if count % backpropagation == 0:#每进行这些轮预测，将lOSS值相加，才进行一次反向传播。如果每轮都进行一次反向传播，训练loss非常不稳定。
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                totalloss =totalloss+loss
                loss = 0
        acc,verifyLoss = verify(model,verifyData,loss_function)#计算一下验证集准确率和损失值
        print(f'epoch:{i:3} trainLoss:{totalloss:10.8f} verifyLoss:{verifyLoss} verifyacc:{acc}')#输出训练集整体损失值、验证集损失值和验证集准确率
        if acc>finalacc:#如果验证集准确率高于之前，保存模型，命名为模型名称+acc.pt
            finalacc = acc
            print("准确率较高，模型保存")
            torch.save(model.state_dict(), modelFileOut+'acc.pt')
        if verifyLoss<finalVerifyLoss:#如果验证集LOSS值低于之前，保存模型，命名为模型名称+loss.pt
            finalVerifyLoss = verifyLoss
            print("loss较低，模型保存")
            torch.save(model.state_dict(), modelFileOut+'loss.pt')
    torch.save(model.state_dict(), modelFileOut + 'finalloss.pt') #最后保存一个训练集最终趋于收敛的模型
    return trainseqlist,trainlablelist #返回训练集结果和训练集标签
def verify(model,verifyData,loss_function):#验证集验证和准确率计算
    pretest = list()#验证集结果列表
    targettest = list()#验证集标签列表
    loss = 0#损失值初始化
    for j in range(len(verifyData) - 4):#验证集测试
        with torch.no_grad():
            seq, lables = getdata(verifyData, j)
            y_predtest = model(seq)
            y_predtest = y_predtest.reshape(1, -1)
            single_loss = loss_function(y_predtest, lables)
            loss = loss+single_loss
            if torch.cuda.is_available():
                y_predtest = y_predtest.cpu()
                lables = lables.cpu()
            pretest.append(y_predtest.tolist())#预测值加入列表
            targettest.append(lables.tolist())#标签加入列表
    finalpretest = np.array(pretest).reshape(-1, 1)#调整一下形状
    finaltarget = np.array(targettest).reshape(-1, 1)
    scalar = joblib.load('./scalarFile/power_scalar'+windnumber) #加载功率归一化文件
    finalpretest = scalar.inverse_transform(finalpretest)#反向归一化
    finaltarget = scalar.inverse_transform(finaltarget)#反向归一化
    TotalE = 0
    #准确率计算
    days = len(verifyData)//96
    for z in range(days):
        E = 0
        for m in range(96):
            E = E + (finaltarget[z * 96 + m] - finalpretest[z * 96 + m]) ** 2
        P = 1 - ((E / (maxpower * maxpower)) / 96) ** 0.5
        TotalE = TotalE + P
    return TotalE/days,loss
#可视化训练集数据
def showTrainData(trainseqlist,trainlablelist):
    finalpretrain = np.array(trainseqlist).reshape(-1,1)
    finaltargettrain = np.array(trainlablelist).reshape(-1,1)
    #加载归一化文件，并反向归一化
    scalar = joblib.load('./scalarFile/power_scalar'+windnumber)
    finalpretrain = scalar.inverse_transform(finalpretrain)
    finaltargettrain = scalar.inverse_transform(finaltargettrain)
    plt.ylabel('power')
    plt.xlabel('time')
    plt.grid(True)
    plt.autoscale(axis='x',tight=True)
    plt.plot(finalpretrain,label='predict power')
    plt.plot(finaltargettrain,label='real power')
    plt.legend()
    plt.show()
def shortTermtest(modelFile,trainData): #测试集预测，参数：模型文件位置，测试集数据
    pretest = list()
    targettest = list()
    # 初始化模型并加载模型文件
    model = model_pkg.getModel()
    model.load_state_dict(torch.load(modelFile))
    # 对测试集数据进行预测
    for j in range(len(trainData) - 3):
        with torch.no_grad():
            seq,lables=getdata(trainData,j)
            y_predtest = model(seq)
            if torch.cuda.is_available():
                y_predtest = y_predtest.cpu()
                lables = lables.cpu()
            pretest.append(y_predtest.tolist())
            targettest.append(lables.tolist())
    finalpretest = np.array(pretest).reshape(-1,1)
    finaltarget = np.array(targettest).reshape(-1,1)
    scalar = joblib.load('./scalarFile/power_scalar'+windnumber)
    finalpretest = scalar.inverse_transform(finalpretest)
    finaltarget = scalar.inverse_transform(finaltarget)
    #根据装机容量，比这个预测值大的都削平为最大功率，比0小的都认为是0
    for i in range(len(finalpretest)):
        if finalpretest[i] > maxpower:
            finalpretest[i]=maxpower
        if finalpretest[i]<0:
            finalpretest[i]=0
    return finalpretest,finaltarget
#测试集准确率验证以及数据可视化
def showTestData(pretest,targettest,days):
    TotalE = 0
    for z in range(days):
        E = 0
        for m in range(96):
            E = E + (targettest[z * 96 + m] - pretest[z * 96 + m]) ** 2
        P = 1 - ((E / (maxpower * maxpower)) / 96) ** 0.5
        TotalE = TotalE + P
    # E = 0
    # for m in range(96):
    #     E = E + (targettest[m] - pretest[m]) ** 2
    # P = 1 - ((E / (94.6 * 94.6)) / 96) ** 0.5
    print(days)
    print("月均准确率：", TotalE/days)
    plt.ylabel('power')
    plt.xlabel('time')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    plt.plot(pretest, label='predict power')
    plt.plot(targettest, label='real power')
    plt.legend()
    plt.show()
#将测试集结果写入文件
def writeToFile(date,pretest,targettest,filename):
    df0 = pd.DataFrame(date)
    df1 = pd.DataFrame(pretest)
    df2 = pd.DataFrame(targettest)
    df = pd.concat([df0,df1, df2], axis=1)
    df.columns=["时间","实际功率","预测功率"]
    df.to_excel(filename, index=False, sheet_name='对比数据')




