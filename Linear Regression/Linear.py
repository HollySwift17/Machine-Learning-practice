import numpy as np
import pandas as pd
import os
import copy
import math
import timeit
import datetime



basepath=os.path.abspath(os.path.dirname(__file__))
path=basepath+'/housing.txt'

a=np.loadtxt(path)

A=np.mat(a)


row=A.shape[0]
col=A.shape[1]

rate=0.05
average=[]
print("rate")
print(rate)

trainingTimes=100000
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



def bgd(x,y,init_theta,times,rate):
    
    theta = init_theta
    m = x.shape[0]
    starttime = datetime.datetime.now()
    for i in range(times):
        predict=np.zeros((row-1,1))
        
        for j in range(row-1):
            predict[j,0]=x[j,:].dot(theta)
        
        grad = x.T.dot((predict - y)) / m * rate
        theta -= grad

        """
        提高效率可以这么写：
        predict = x.dot(theta) 
        grad = x.T.dot((predict - y)) / m * rate
        theta -= grad
        """
    endtime = datetime.datetime.now()
    print (endtime - starttime).seconds
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


    theta=bgd(X,Y,theta.copy(),trainingTimes,rate)

    
    pred=testX*theta
    print("testY")
    print(testY*average[13])
    print("pred")
    print(pred*average[13])

    RMSE=RMSE+pow((pred[0,0]*average[13]-testY[0,0]*average[13]),2)

  
   

RMSE=math.sqrt(RMSE/row)
print("RMSE")
print(RMSE)


    
