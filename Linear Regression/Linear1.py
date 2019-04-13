import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import math
import random
import datetime

starttime = datetime.datetime.now()

basepath=os.path.abspath(os.path.dirname(__file__))
path=basepath+'/housing.txt'

a=np.loadtxt(path)

A=np.mat(a)


row=A.shape[0]
col=A.shape[1]

rate=0.008
average=[]
print("rate")
print(rate)

trainingTimes=500000
print("trainingTimes") 
print(trainingTimes)

RMSE=0


def adjust(X):
    #特征缩放
    temp=X
    average1=np.sum(temp)/row
    X=temp/average1
    average.append(average1)
    return X

# 随机梯度下降

def sgd(x, y, init_theta, times, rate):
    
    theta=init_theta
    
    for i in range(times):
        j=random.randint(0,row-2)
        predict = x[j,:].dot(theta) 
        grad = x[j,:].T.dot((predict - y[j,:])) * rate
        theta -= grad 
    endtime = datetime.datetime.now()
    return theta



for i in range(col):
    A[:,i]=adjust(A[:,i])
    


for testIndex in range(row):
   
    #留一法,对每一行数据进行训练验证
    theta=np.zeros((col,1))
    
    
    X=copy.copy(A[:,:])
    for i in range(row):
        X[i,col-1]=1

    Y=copy.copy(A[:,col-1])

    testX=X[testIndex,:]
    testY=Y[testIndex,:]
    X=np.delete(X, testIndex, 0)
    Y=np.delete(Y, testIndex, 0)


    theta=sgd(X,Y,theta.copy(),trainingTimes,rate)

    
    pred=testX*theta
    print("testY")
    print(testY*average[13])
    print("pred")
    print(pred*average[13])

    RMSE=RMSE+pow((pred[0,0]*average[13]-testY[0,0]*average[13]),2)
    


RMSE=math.sqrt(RMSE/row)
print("RMSE")
print(RMSE)

