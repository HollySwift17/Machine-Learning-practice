import scipy.io as scio
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
import numpy as np
import copy
from sklearn import preprocessing
import heapq
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split


data = scio.loadmat('bibtex.mat')
temp=list(data.values())
X=np.mat(temp[3])
#print(X)
X=np.mat(preprocessing.scale(X))#标准化(Standardization)
Y=np.transpose(np.mat(temp[4]))
#按行打乱

per = np.random.permutation(X.shape[0])		#打乱后的行号
X = X[per, :]		#获取打乱后的训练数据
Y = Y[per]

X=X[:int(0.3*X.shape[0]),:]
Y=Y[:int(0.3*Y.shape[0]),:]

X_train,X_test,y_train,y_test= train_test_split(X,Y,test_size=0.2,random_state=1)
X_train,X_validation,y_train,y_validation= train_test_split(X_train,y_train,test_size=0.2,random_state=1)


n=X_train.shape[0]
d=X_train.shape[1]
c=y_train.shape[1]
F=copy.deepcopy(y_train)
W=np.random.rand(d,c)
S=np.eye(n)
k=1

while 1:
    D=np.eye(d)
    for j in range(d):
        D[j,j]=np.linalg.norm(W[j,:],ord=2) 
    m=np.dot(np.dot(np.ones((1,n)),S),np.ones((n,1)))[0][0]

    H=np.eye(n)-1/m*np.dot(np.ones((n,1)),np.dot(np.ones((1,n)),S))
    temp1=np.dot(X_train.T,H)
    temp2=np.dot(temp1,S)
    temp3=np.dot(temp2,H)
    temp4=np.dot(temp3,X_train)
    temp5=(temp4+k*np.mat(D)).I
    temp6=np.dot(temp5,temp3)
    temp7=np.dot(temp6,y_train)
           
    W_=temp7

    temp8=np.dot(np.dot(F.T,S),np.ones((n,1)))
    temp9=np.dot(np.dot(np.dot(W.T,X_train.T),S),np.ones((n,1)))
    b=1/m*(temp8-temp9)
            
    F_=np.dot(X_train,W)+np.dot(np.zeros((n,1)),b.T)
            

    for i in range(F_.shape[0]):
        for j in range(F_.shape[1]):
            if F_[i,j]<=0:
                F_[i,j]=0
            if F_[i,j]>=1:
                F_[i,j]=1
            
    
    t=np.linalg.norm(W-W_)
    t_=np.linalg.norm(W)
    

    if t/t_<0.001:
        break
    print(t/t_)


    W=W_
    F=F_



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





rates=list(range(1,100))
for i in range(len(rates)):
    rates[i]=float(rates[i])/float(100)

precisions=[]
recalls=[]
for rate in rates:
    print(rates)
    score_list=[]
    l=[]
    for j in range(d):
        l.append(np.linalg.norm(W[j,:],ord=2))

    max_num_index_list = map(l.index, heapq.nlargest(round(rate*d), l))
    index_list=list(max_num_index_list)

    for each in index_list:
        each=each-1
            

    X_train_=X_train[:,index_list]
    X_validation_=X_validation[:,index_list]

    print(index_list)

    clf=BinaryRelevance(SGDClassifier(loss="hinge",penalty="l2")).fit(X_train_,y_train)
    pred=clf.predict(X_validation_).toarray()

    precision, recall=cal_precision_and_recall(y_validation,pred)
    print(precision)
    print(recall)
    
    precisions.append(precision)
    recalls.append(recall)


print(precisions)
print(recalls)


import matplotlib.pyplot as plt
plt.figure()
plt.plot(rates,precisions)
plt.show()


