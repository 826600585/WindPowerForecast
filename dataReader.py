from sklearn import preprocessing
import numpy as np
import joblib
import pandas as pd
import copy
import torch
from common import function
from torch.utils.data import Dataset
from torch.autograd import Variable
#训练集数据读取
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
class trainDataset(Dataset):
    def __init__(self):
        # 读取训练集数据表格
        windfarm_Data = pd.read_excel(trainfilepath, sheet_name=trainsheetname, skiprows=trainskiprow, nrows=trainuserow,usecols=[i for i in range(1, 22)])
        # 将dataframe转为numpy
        windfarm_DataNP = windfarm_Data.values
        windfarm_DataNP = windfarm_DataNP.astype(np.float32)
        # 初始化功率min-max归一化函数
        powerScaler = preprocessing.MinMaxScaler()
        # 归一化第一列功率数据
        windfarm_DataNP[:, 0] = powerScaler.fit_transform(windfarm_DataNP[:, 0].reshape(-1, 1)).reshape(1, -1)
        # 保存功率归一化参数文件
        joblib.dump(powerScaler, './scalarFile/power_scalar' + windnumber)
        # sin归一化风向数据
        for i in range(1, 8):
            windfarm_DataNP[:, i] = np.sin(windfarm_DataNP[:, i] / 180 * np.pi)
        # 初始化除风向外其他特征min-max归一化函数
        featureScaler = preprocessing.MinMaxScaler()
        # 对其他特征进行归一化
        windfarm_DataNP[:, 8:] = featureScaler.fit_transform(windfarm_DataNP[:, 8:])
        # 保存特征归一化参数文件
        joblib.dump(featureScaler, './scalarFile/Feature_scalar' + windnumber)
        self.feature_data = windfarm_DataNP[:, 1:]
        self.power_data = windfarm_DataNP[:, 0]
        self.len = windfarm_DataNP.shape[0]-3

    def __getitem__(self, index):
        return self.feature_data[index:index+4,:], self.power_data[index+3,0]
    def __len__(self):
        return self.len
class testDataset(Dataset):
    def __init__(self):
        windfarm_Data = pd.read_excel(testfilepath, sheet_name=testsheetname, skiprows=testskiprow, nrows=testuserow,usecols=[i for i in range(1, 22)])
        windfarm_DataNP = windfarm_Data.values
        windfarm_DataNP = windfarm_DataNP.astype(np.float32)
        powerScaler = joblib.load('./scalarFile/power_scalar' + windnumber)
        windfarm_DataNP[:, 0] = powerScaler.transform(windfarm_DataNP[:, 0].reshape(-1, 1)).reshape(1, -1)
        for i in range(1, 8):
            windfarm_DataNP[:, i] = np.sin(windfarm_DataNP[:, i] / 180 * np.pi)
        featureScaler = joblib.load('./scalarFile/Feature_scalar' + windnumber)
        windfarm_DataNP[:, 8:] = featureScaler.transform(windfarm_DataNP[:, 8:])
        self.feature_data = windfarm_DataNP[:, 1:]
        self.power_data = windfarm_DataNP[:, 0]
        self.len = windfarm_DataNP.shape[0]-3
    def __getitem__(self, index):
        return self.feature_data[index:index+4,:], self.power_data[index+3,0]
    def __len__(self):
        return self.len
def testDateReader(link,sheetname,skiprow,nrows):
    windfarm_Date = pd.read_excel(link, sheet_name=sheetname, skiprows=skiprow, nrows=nrows,usecols=[0,0])
    return windfarm_Date
#单批数据读取函数，每次从数据集读取
def getdata(data,idm):
    # 每次读取训练集或测试集，以idm为开始，向后的4个时间序列数据，标签是最后一个时间序列
    seq,lables=copy.deepcopy(data[idm:idm+4,1:21]),copy.deepcopy(data[idm+3,0])
    lables = lables.reshape(-1, 1)
    seq = torch.from_numpy(seq)
    lables=torch.from_numpy(lables)
    if torch.cuda.is_available():
        seq = seq.cuda()
        lables = lables.cuda()
    return seq,lables
