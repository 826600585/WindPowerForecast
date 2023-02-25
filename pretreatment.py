import pandas as pd
import numpy as np
import math
import pywt
import matplotlib.pyplot as plt
from scipy import interpolate

#将同一个EXCEL不同sheet进行合并的代码
def dataMerge(datalink):
    df1 = pd.read_excel(datalink,sheet_name='实际功率',usecols=[0,2])
    df2 = pd.read_excel(datalink,sheet_name='气象数据',usecols=[1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,21])
    df3 = pd.merge(df1,df2,on='时间',how='right')
    return df3

#创建时间序列新统计特征
def createNewFeature(datalink,dataFrame):
    #上一时刻风速统计特征
    dfcreate2 = pd.read_excel(datalink,sheet_name='气象数据',usecols=[12])
    dfcreate3 = pd.DataFrame(['0'], columns=dfcreate2.columns)
    dfcreate4 = pd.concat([dfcreate3, dfcreate2])
    dfcreate4 = dfcreate4.reset_index(drop=True)
    #新增加特征列名设置
    dfcreate4.columns = ['上一时刻110米风速']
    dfcreate6 = pd.DataFrame(columns=['110米近风速3个mean'])
    dfcreate61 = pd.DataFrame(columns=['110米近1对近3风速趋势'])
    #风速平均值及趋势计算
    for i in range(len(dfcreate2)):
        #风速趋势计算
        if i < 3:
            dfcreate61.loc[i] = 0
        else:
            dfcreate61.loc[i] = dfcreate2.loc[i - 1].values / dfcreate2.loc[i - 3].values
        #风速平均值计算
        if i < 2:
            dfcreate6.loc[i] = 0
        else:
            dfcreate6.loc[i] = (dfcreate2.loc[i].values + dfcreate2.loc[i - 1].values + dfcreate2.loc[i - 2].values) / 3
    #新特征列与原特征列合并
    dataFrame = pd.concat([dataFrame, dfcreate4, dfcreate6, dfcreate61], axis=1)
    return dataFrame

#缺失及异常数据的处理
def abnormalData(dataFrame,MaxPower):
    delelist=[]
    #遍历一遍数据
    for i in range(len(dataFrame)):
        #实际功率小于0的数据均赋值为-1（异常值标记）
        if dataFrame.loc[i,'实际功率']<0:
            dataFrame.loc[i,'实际功率']=-1
        #连续时间相同实际功率（不为0，不是额定最大功率），不符合实际认知逻辑，将index添加到待删除列表。
        if i != len(dataFrame)-1 and dataFrame.loc[i+1,'实际功率']!=0 and dataFrame.loc[i+1,'实际功率']!= MaxPower:
            if dataFrame.loc[i+1,'实际功率']-dataFrame.loc[i,'实际功率'] == 0:
                delelist.append(i+1)
        #连续长时间为0，赋值为-1（异常值标记）
        if dataFrame.loc[i,'实际功率']==0:
            zeroCount = 1
            while True:
                if dataFrame.loc[i+zeroCount,'实际功率'] == 0:
                    zeroCount = zeroCount+1
                if dataFrame.loc[i+zeroCount,'实际功率'] != 0:
                    break
            #设置连续30个时间点均为0时，标记为-1（异常值）
            if zeroCount>30:
                for q in range(1,zeroCount):
                    dataFrame.loc[i+q, '实际功率'] = -1
    #将待删除列表中的值标记为-1
    for m in delelist:
        dataFrame.loc[m, '实际功率'] = -1
    #异常值、缺失值填充（少部分缺失、异常，使用线性差值填充）
    dataFrame['实际功率'].replace(-1,np.nan,inplace=True)
    dataFrame['实际功率']=dataFrame['实际功率'].interpolate(method = 'linear',axis = 0,limit_direction = 'forward',limit = 3)
    dataFrame['实际功率'].replace(np.nan,-1,inplace=True)
    #异常值、缺失值填充（大部分缺失，采用直接删除法）
    for i in range(len(dataFrame)):
        if dataFrame.loc[i,'实际功率'] == -1:
            dataFrame = dataFrame.drop([i])
    return dataFrame

def sgn(num):
    if(num > 0.0):
        return 1.0
    elif(num == 0.0):
        return 0.0
    else:
        return -1.0

def wavelet_noising(new_df):
    data = new_df
    data = data.values.T.tolist()  # 将np.ndarray()转为列表
    w = pywt.Wavelet('db4')#选择db4小波基
    [ca3, cd3, cd2, cd1] = pywt.wavedec(data, w, level=3)  # 3层小波分解

    length1 = len(cd1)
    length0 = len(data)

    Cd1 = np.array(cd1)
    abs_cd1 = np.abs(Cd1)
    median_cd1 = np.median(abs_cd1)

    sigma = (1.0 / 0.6745) * median_cd1
    lamda = sigma * math.sqrt(2.0 * math.log(float(length0 ), math.e))#自适应阈值
    usecoeffs = []
    usecoeffs.append(ca3)  # 向列表末尾添加对象

    #软阈值方法
    a = 1

    for k in range(length1):
        if (abs(cd1[k]) >= lamda):
            cd1[k] = sgn(cd1[k]) * (abs(cd1[k]) - a * lamda)
        else:
            cd1[k] = 0.0

    length2 = len(cd2)
    for k in range(length2):
        if (abs(cd2[k]) >= lamda):
            cd2[k] = sgn(cd2[k]) * (abs(cd2[k]) - a * lamda)
        else:
            cd2[k] = 0.0

    length3 = len(cd3)
    for k in range(length3):
        if (abs(cd3[k]) >= lamda):
            cd3[k] = sgn(cd3[k]) * (abs(cd3[k]) - a * lamda)
        else:
            cd3[k] = 0.0
    usecoeffs.append(cd3)
    usecoeffs.append(cd2)
    usecoeffs.append(cd1)
    recoeffs = pywt.waverec(usecoeffs, w)#信号重构
    return recoeffs


datalink = './data/J00154_origin.xlsx'
print("数据合并....")
df3 = dataMerge(datalink)
print("数据合并完成")
print("创建新统计特征....")
df3 = createNewFeature(datalink,df3)
print("统计特征创建完成")
print("异常、缺失数据处理....")
df3 = abnormalData(df3,94.6)
print("异常、缺失数据处理完成")
df3 = df3.reset_index(drop=True)
# print("数据降噪....")
# dfpower = df3.iloc[:,1]
# data_denoising = wavelet_noising(dfpower)
# df3.iloc[:,1]=data_denoising
# plt.plot(df3.iloc[:,1],label='power2')
# plt.show()
print("写入Excel...")
df3.to_excel('./data/J00154_Merge2.xlsx',index=False,sheet_name='汇总')
print("数据预处理完成")