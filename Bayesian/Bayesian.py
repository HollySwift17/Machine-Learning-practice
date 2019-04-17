import numpy as np
import pandas as pd
import os
import copy
import math


basepath=os.path.abspath(os.path.dirname(__file__))
path=basepath+'/wine.txt'

a=np.loadtxt(path)
A=np.mat(a)


row=A.shape[0]
col=A.shape[1]
esp=0.000001
correct=0
confusion=np.zeros((3,3))
alpha=1
reality1=[]
pred1=[]
score1=[]
reality2=[]
pred2=[]
score2=[]
reality3=[]
pred3=[]
score3=[]

row1=np.sum(A[:,0]==1)
row2=np.sum(A[:,0]==2)
row3=np.sum(A[:,0]==3)


"""

from sklearn.naive_bayes import GaussianNB

for i in range(row):
    X=A[:,1:14].copy()
    Y=A[:,0].copy()

    testX=X[i,:].copy()
    testY=Y[i,:].copy()

    X=np.delete(X,i,0)
    Y=np.delete(Y,i,0)
    clf = GaussianNB()
        
    clf.fit(X,Y)
    pred=clf.predict(testX)
    if abs(pred-testY)<esp:
        correct=correct+1


"""

def cal_para(x):
    mean=np.zeros((1,col-1))
    for i in range(col-1):
        mean[0,i]=np.mean(x[:,i])    
    cov=np.cov(x.T)
    return mean,cov



def cal_density(x,mean,cov,n=col-1):  
    #求概率密度
    det_cov=np.linalg.det(cov)
    cov_=np.linalg.inv(cov)
    para=1/(pow((2*np.pi),n/2)*pow(det_cov,0.5))
    exponent=-0.5*(x-mean).dot(cov_).dot((x-mean).T)
    return para*pow(np.e,(exponent[0,0]))




def classify(trainData, labels, features):

    row1=np.sum(labels==1)
    row2=np.sum(labels==2)
    row3=np.sum(labels==3)
    
    P_y = {}
    P_y[1]=row1/labels.shape[0]
    P_y[2]=row2/labels.shape[0]
    P_y[3]=row3/labels.shape[0]
    

    #对label==1,2,3分别求均值和协方差矩阵

    X1=trainData[0:row1,:]
    X2=trainData[row1:row1+row2,:]
    X3=trainData[row1+row2:row1+row2+row3,:]
    
    mean1,cov1=cal_para(X1)
    mean2,cov2=cal_para(X2)
    mean3,cov3=cal_para(X3)

    #条件概率
    P_xy={}
    P_xy[1]=cal_density(features,mean1,cov1)
    P_xy[2]=cal_density(features,mean2,cov2)
    P_xy[3]=cal_density(features,mean3,cov3)

    P={}
    P[1]=P_xy[1]*P_y[1]
    P[2]=P_xy[2]*P_y[2]
    P[3]=P_xy[3]*P_y[3]

    summ=P[1]+P[2]+P[3]
    P[1]=P[1]/summ
    P[2]=P[2]/summ
    P[3]=P[3]/summ

    pred=max(P, key=P.get)
    return pred,P[pred]  #概率最大值对应的类别



for i in range(10):
    curr1=range(int(i*row1/10),int((i+1)*row1/10))
    curr2=range(row1+int(i*row2/10),row1+int((i+1)*row2/10))
    curr3=range(row1+row2+int(i*row3/10),row1+row2+int((i+1)*row3/10))

    #curr=curr1+curr2+curr3
    
    X=A[:,1:14].copy()
    Y=A[:,0].copy()

    testX1=X[curr1,:].copy()
    testY1=Y[curr1,:].copy()

    testX2=X[curr2,:].copy()
    testY2=Y[curr2,:].copy()

    testX3=X[curr3,:].copy()
    testY3=Y[curr3,:].copy()

    X=np.delete(X,curr3,0)
    X=np.delete(X,curr2,0)
    X=np.delete(X,curr1,0)
    
    Y=np.delete(Y,curr3,0)
    Y=np.delete(Y,curr2,0)
    Y=np.delete(Y,curr1,0)

        
    for j in curr1:
        _pred,_score=classify(X,Y,A[j,1:14])
        
        if _pred==1:
            correct=correct+1

        reality1.append(1)
        pred1.append(_pred)
        score1.append(_score)
        confusion[0,int(_pred)-1]=confusion[0,int(_pred)-1]+1

        print(1 ,"——",_pred)

    for j in curr2:
        _pred,_score=classify(X,Y,A[j,1:14])
        
        if _pred==2:
            correct=correct+1

        reality2.append(2)
        pred2.append(_pred)
        score2.append(_score)
        confusion[1,int(_pred)-1]=confusion[1,int(_pred)-1]+1

        print(2 ,"——",_pred)
        
    for j in curr3:
        _pred,_score=classify(X,Y,A[j,1:14])
        
        if _pred==3:
            correct=correct+1

        reality3.append(3)
        pred3.append(_pred)
        score3.append(_score)
        confusion[2,int(_pred)-1]=confusion[2,int(_pred)-1]+1

        print(3 ,"——",_pred)



accuracy=correct/row
precision={}
precision[1]=confusion[0,0]/np.sum(confusion[0,:])
precision[2]=confusion[1,1]/np.sum(confusion[1,:])
precision[3]=confusion[2,2]/np.sum(confusion[2,:])
recall={}
recall[1]=confusion[0,0]/np.sum(confusion[:,0])
recall[2]=confusion[1,1]/np.sum(confusion[:,1])
recall[3]=confusion[2,2]/np.sum(confusion[:,2])
F={}
F[1]=((pow(alpha,2)+1)*precision[1]*recall[1])/(pow(alpha,2)*(precision[1]+recall[1]))
F[2]=((pow(alpha,2)+1)*precision[2]*recall[2])/(pow(alpha,2)*(precision[2]+recall[2]))
F[3]=((pow(alpha,2)+1)*precision[3]*recall[3])/(pow(alpha,2)*(precision[3]+recall[3]))



#ROC曲线绘制

import matplotlib.pyplot as plt

def draw_roc(reality,pred,score,row,color):
    temp=np.zeros((row,3))
    temp[:,0]=np.array(reality)
    temp[:,1]=np.array(pred)
    temp[:,2]=np.array(score)

    sorted_temp=temp[np.lexsort(-temp.T)] 

    curr_x=0
    curr_y=0

    node_x=[]
    node_y=[]

    for i in range(row):
        if sorted_temp[i,0]!=sorted_temp[i,1] :
            curr_x=curr_x+1
        else:
            curr_y=curr_y+1
 
        node_x.append(curr_x)
        node_y.append(curr_y)

    node_x=np.array(node_x)/curr_x
    node_y=np.array(node_y)/curr_y

    plt.plot(node_x,node_y,c=color)

    return np.trapz(node_y,node_x)



auc1=draw_roc(reality1,pred1,score1,row1,'b')
auc2=draw_roc(reality2,pred2,score2,row2,'g')
auc3=draw_roc(reality3,pred3,score3,row3,'r')

plt.show()
