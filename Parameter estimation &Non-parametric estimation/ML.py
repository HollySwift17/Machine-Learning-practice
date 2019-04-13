import numpy as np
import pandas as pd
import os
import copy
import math
from scipy import integrate 


basepath=os.path.abspath(os.path.dirname(__file__))
path=basepath+'/HWData3.txt'

a=np.loadtxt(path)
A=np.mat(a)


row=A.shape[0]
col=A.shape[1]
esp=0.000001
correct=0



def cal_para(x):
    mean=np.zeros((1,col-1))
    for i in range(col-1):
        mean[0,i]=np.mean(x[:,i])
    
    cov=np.cov(x.T)
    return mean,cov







def cal_density(x1,x2,x3,x4,mean,cov,n=4):
    #求概率密度
    x=np.zeros((1,col-1))
    x[0,0]=x1
    x[0,1]=x2
    x[0,2]=x3
    x[0,3]=x4
    det_cov=np.linalg.det(cov)
    cov_=np.linalg.inv(cov)
    para=1/(pow((2*np.pi),n/2)*pow(det_cov,0.5))
    exponent=-0.5*(x-mean).dot(cov_).dot((x-mean).T)

    return para*pow(np.e,(exponent[0,0]))
    


def cal_probability(mean,x,cov,n=4):
    #根据概率密度求积分的概率
    #积分下限
    a1=x[0,0]-cov[0,0]
    a2=x[0,1]-cov[1,1]
    a3=x[0,2]-cov[2,2]
    a4=x[0,3]-cov[3,3]
    #积分下限
    b1=x[0,0]+cov[0,0]
    b2=x[0,1]+cov[1,1]
    b3=x[0,2]+cov[2,2]
    b4=x[0,3]+cov[3,3]

    return cal_density(x[0,0],x[0,1],x[0,2],x[0,3],mean,cov)
    
    probability=integrate.nquad(cal_density,[[a1,b1],[a2,b2],[a3,b3],[a4,b4]],args=(mean,cov))
    return probability[0]    




for t in range(5):
    #五折交叉检验

    X1=A[0:50,0:4].copy()
    X2=A[50:100,0:4].copy()
    X3=A[100:150,0:4].copy()

    X1=np.delete(X1,[10*t,10*t+1,10*t+2,10*t+3,10*t+4,10*t+5,10*t+6,10*t+7,10*t+8,10*t+9], 0)
    X2=np.delete(X2,[10*t,10*t+1,10*t+2,10*t+3,10*t+4,10*t+5,10*t+6,10*t+7,10*t+8,10*t+9], 0)
    X3=np.delete(X3,[10*t,10*t+1,10*t+2,10*t+3,10*t+4,10*t+5,10*t+6,10*t+7,10*t+8,10*t+9], 0)

    
    test1=A[t*10:t*10+10,0:4].copy()
    test2=A[50+t*10:60+t*10,0:4].copy()
    test3=A[100+t*10:110+t*10,0:4].copy()

    

    mean1,cov1=cal_para(X1)
    mean2,cov2=cal_para(X2)
    mean3,cov3=cal_para(X3)



    #测试
    

    for test in test1:
        p1=cal_probability(mean1,test,cov1)
        p2=cal_probability(mean2,test,cov2)
        p3=cal_probability(mean3,test,cov3)
        print(p1,p2,p3)
        if(p1>p2)and(p1>p3):
            correct=correct+1
        
    for test in test2:
        p1=cal_probability(mean1,test,cov1)
        p2=cal_probability(mean2,test,cov2)
        p3=cal_probability(mean3,test,cov3)
        print(p1,p2,p3)
        if(p2>p1)and(p2>p3):
            correct=correct+1

        
    for test in test3:
        p1=cal_probability(mean1,test,cov1)
        p2=cal_probability(mean2,test,cov2)
        p3=cal_probability(mean3,test,cov3)
        print(p1,p2,p3)
        if(p3>p2)and(p3>p1):
            correct=correct+1
        


print(correct/150)























    
