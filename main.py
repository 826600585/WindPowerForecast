from dataReader import trainDataReader,testDataReader,testDateReader
import train as tn
import torch
import numpy as np
import random
from common import function
#设置随机数种子
modelpath = function.getConfig("Config",'testfile','modelfilepath')
testfilepath = function.getConfig("Config",'testfile','filepath')
trainfilepath = function.getConfig("Config",'trainfile','filepath')
testsheetname = function.getConfig("Config",'testfile','sheetname')
testskiprow = int(function.getConfig("Config",'testfile','skiprow'))
trainskiprow = int(function.getConfig("Config",'trainfile','skiprow'))
testuserow = int(function.getConfig("Config",'testfile','userow'))
trainuserow = int(function.getConfig("Config",'trainfile','userow'))
trainsheetname = function.getConfig("Config",'trainfile','sheetname')
windnumber  = function.getConfig("Config",'trainfile','windnumber')
randomseed  = int(function.getConfig("Config",'model','randomseed'))
trainEpoch = int(function.getConfig("Config",'model','epoch'))

def setup_seed(seed):
    torch.manual_seed( seed )
    torch.cuda.manual_seed_all( seed )
    np.random.seed( seed )
    random.seed( seed )
#开始训练、生成模型文件
def starttrain(link,sheetname,skiprows,nrows,epochs,modelFileOut):#参数：训练集文件路径名、sheet表名、跳过行数、使用行数、训练轮次，模型文件名称。
    setup_seed(randomseed)
    trainDataLink = link
    trainData = trainDataReader(trainDataLink,sheetname,skiprows,nrows)#训练集读取函数
    verifyData = testDataReader(trainDataLink,sheetname,skiprows+nrows,3000)#验证集读取函数（使用训练集最后3000行数据）
    trainResult,traintarget = tn.train(trainData,epochs,modelFileOut,verifyData)#进行训练，输出训练结果和标签
    tn.showTrainData(trainResult,traintarget)#进行训练集数据展示
#使用训练好的模型文件进行训练
def starttest(modelFile,testDatalink,sheetname,skipnrows,nrows):#参数：模型文件，测试集文件路径名、sheet表名、跳过行数、使用行数
    setup_seed(randomseed)
    modelFile = modelFile
    testDatalink = testDatalink
    dataDate= testDateReader(testDatalink,sheetname,skipnrows+3,nrows-3)
    testData = testDataReader(testDatalink,sheetname,skipnrows,nrows)#测试集读取函数
    result,target = tn.shortTermtest(modelFile,testData)#短期结果和标签输出
    tn.showTestData(result,target,nrows//96)#展示测试集训练数据和标签数据，计算准确率
    tn.writeToFile(dataDate,target,result,"./forecastData/"+windnumber+"预测数据.xlsx")#将预测结果和实际标签写入表格

#starttrain(trainfilepath,trainsheetname,trainskiprow,trainuserow,trainEpoch,"./model/model"+windnumber)
starttest(modelpath,testfilepath,testsheetname,skipnrows=int(testskiprow),nrows=int(testuserow))
