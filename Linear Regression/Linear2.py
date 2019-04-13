import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import math
import datetime



basepath=os.path.abspath(os.path.dirname(__file__))
path=basepath+'/housing.txt'

a=np.loadtxt(path)

A=np.mat(a)


row=A.shape[0]
col=A.shape[1]

rate=0.0001
average=[]
print("rate")
print(rate)

trainingTimes=50000
print("trainingTimes") 
print(trainingTimes)

L2=0.01
print("L2")
print(L2)

RMSE=0


def adjust(X):
    #特征缩放
    temp=X
    average1=np.sum(temp)/row
    X=temp/average1
    average.append(average1)
    return X


def ridge(x,y,init_theta,times,crete,L2):

    theta = init_theta.copy() 
        
    for j in range(times):
       
        pred = np.dot(x,theta)
        err = pred-y
        
        for i in range(len(theta)): 
            theta[i] = (1-2*rate*L2)*theta[i] - 2*rate*np.dot(err.T,x[:,i])

    print(theta)
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


    theta=ridge(X,Y,theta.copy(),trainingTimes,rate,L2)

    
    pred=testX*theta
    print("testY")
    print(testY*average[13])
    print("pred")
    print(pred*average[13])

    RMSE=RMSE+pow((pred[0,0]*average[13]-testY[0,0]*average[13]),2)




RMSE=math.sqrt(RMSE/row)
print("RMSE")
print(RMSE)

    
