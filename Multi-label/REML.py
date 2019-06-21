import scipy.io as scio
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
import numpy as np
import copy
from sklearn import preprocessing
import heapq
from sklearn.model_selection import train_test_split


data = scio.loadmat('bibtex.mat')
temp=list(data.values())
X=np.mat(temp[3])
#print(X)
X=np.mat(preprocessing.scale(X))#标准化(Standardization)
Y=np.transpose(np.mat(temp[4]))
#按行打乱
per = np.random.permutation(X.shape[0])#打乱后的行号
X = X[per, :]#获取打乱后的训练数据
Y = Y[per]

#X=X[:int(0.3*X.shape[0]),:]
#Y=Y[:int(0.3*Y.shape[0]),:]

X_train,X_test,y_train,y_test= train_test_split(X,Y,test_size=0.2,random_state=1)
X_train,X_validation,y_train,y_validation= train_test_split(X_train,y_train,test_size=0.2,random_state=1)

#———————————手动分割线————————————

n=X_train.shape[0]
d=X_train.shape[1]
c=y_train.shape[1]

k=5
W=np.mat(np.random.rand(d,c))
U=np.mat(np.random.rand(d,k))
V=np.mat(np.random.rand(k,c))
H=np.mat(np.random.rand(d,c))

lanmuda1=0.03
lanmuda2=0.3
lanmuda3=0.01

rou=0.1


miu=1/float(rou)*y_train
Z=np.dot(X_train,H)

def cal_precision_and_recall(y,pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            if pred[i,j]==1 and y[i,j] == 1:
                TP += 1
            if pred[i,j]==1 and y[i,j] == 0:
                FP += 1
            if pred[i,j]==0 and y[i,j] == 0:
                TN += 1
            if pred[i,j]==0 and y[i,j] == 1:
                FN += 1            
    recall = float(TP)/float(TP+FN)
    precision = float(TP)/(TP+FP)    
    return precision, recall


for time in range(100):
    #———————————手动分割线————————————
    temp=np.dot(X_train,H)
    for j in range(c):
        #print(j)
        t1=lanmuda2*np.eye(k)+np.sum(np.dot(np.dot(X_train[i,:],U).T,np.dot(X_train[i,:],U))for i in range(n))
        t2=np.sum(np.dot(y_train[i,j]-temp[i,j],np.dot(X_train[i,:],U).T)for i in range(n))        
        V[:,j]=np.dot(np.linalg.pinv(t1),t2)
    print("done")

   #———————————手动分割线————————————
    t1=lanmuda2*np.mat(np.mat(np.eye(1)))
    t2=np.mat(np.zeros((1,d*k)))
    tt=np.dot(X_train,H)
    for i in range(n):
        #print(i)
        for j in range(c):
            temp1=np.dot(X_train[i,:].T,V[:,j].T)
            temp2=temp1.reshape((1,d*k))
            temp3=np.dot(temp2,temp2.T)
            t1+=temp3
            temp4=np.dot(y_train[i,j]-tt[i,j],temp2)
            t2+=temp4
            
    u=np.dot(np.linalg.pinv(t1),t2)
    U=u.reshape((d,k))
   
    #———————————手动分割线————————————
    
    A=np.dot(X_train,H)
    def soft(a,b):
        result=np.zeros((a.shape[0],a.shape[1]))
        for i in range(a.shape[0]):
            if a[i,0]==0:
                sign=0
            else:
                if a[i,0]>0:
                    sign=1
                else:
                    sign=-1
            result[i,0]=sign*(max(abs(a[i,0])-b,0))
        return result
    for j in range(c):
        Z[:,j]=soft(np.dot(X_train,H[:,j])+miu[:,j],float(lanmuda3)/float(rou))
                    
    t1=lanmuda1*np.eye(d)+(rou+1)*np.dot(X_train.T,X_train)
    t2=y_train-np.dot(np.dot(X_train,U),V)
    H=np.dot(np.linalg.pinv(t1),(np.dot(X_train.T,t2)+rou*np.dot(X_train.T,Z-miu)))
            
    miu += np.dot(X_train, H) - Z
    
    if time!=0:
        cost_=cost
    cost=np.linalg.norm(y_train-np.dot(np.dot(X_train,U),V)-np.dot(X_train,H))+lanmuda1*np.linalg.norm(H)+lanmuda2*(np.linalg.norm(U)+np.linalg.norm(V))+lanmuda3*np.linalg.norm(np.dot(X,H))
    print(cost)
    
    if time!=0 and abs(cost-cost_)/cost_<0.001:
            break

    print("——————————————————")


W=np.dot(U,V)

pred=np.dot(X_validation,W)+np.dot(X_validation,H)
print(pred)

for i in range(pred.shape[0]):
    for j in range(pred.shape[1]):
        if pred[i,j]>=0.5:
            pred[i,j]=1
        else:
            pred[i,j]=0
precision, recall=cal_precision_and_recall(y_validation,pred)
print(precision)
print(recall)
