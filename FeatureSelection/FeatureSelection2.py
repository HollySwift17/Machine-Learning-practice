import scipy.io as scio
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
import numpy as np
import copy
from sklearn import preprocessing
import heapq
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import random

data = scio.loadmat('emotions.mat')
temp=list(data.values())
X=np.mat(temp[3])
X=np.mat(preprocessing.scale(X))#标准化(Standardization)
Y=np.transpose(np.mat(temp[4]))
#按行打乱

per = np.random.permutation(X.shape[0])		#打乱后的行号
X = X[per, :]		#获取打乱后的训练数据
Y = Y[per]




rates=[float(1)/float(6),float(2)/float(6),float(3)/float(6),
       float(4)/float(6),float(5)/float(6),float(6)/float(6)]
k=1
rate1=0.5
scores=[]
for rate in rates:
    score_list=[]
    for i in range(100):
        X_train,X_test,y_train,y_test= train_test_split(X,Y,test_size=0.2,random_state=1)
        X_train,X_validation,y_train,y_validation= train_test_split(X_train,y_train,test_size=0.2,random_state=1)


        n=X_train.shape[0]
        d=X_train.shape[1]
        c=y_train.shape[1]
        F=copy.deepcopy(y_train)
        F[int(rate1*F.shape[0]):,:]=0
        W=np.random.rand(d,c)
        S=np.eye(n)
        S[int(rate1*S.shape[0]):,:]=0

        for i in range(100):
            #print(i)
            D=np.eye(d)
            for j in range(d):
                D[j,j]=np.linalg.norm(W[j,:],ord=2) 
            m=np.dot(np.dot(np.ones((1,n)),S),np.ones((n,1)))[0][0]

            H=np.eye(n)-1/m*np.dot(np.ones((n,1)),np.dot(np.ones((1,n)),S))
            temp1=np.dot(X_train.T,H)
            temp2=np.dot(temp1,S)
            temp3=np.dot(temp2,H)
            temp4=np.dot(temp3,X_train)
            temp5=np.linalg.pinv((temp4+k*np.mat(D)))#伪逆
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
            
            t=0
            for i in range(d):
                for j in range(c):
                    t+=(W[i,j]-W_[i,j])*(W[i,j]-W_[i,j])
            print(t)
            if t<0.001:
                break

            W=W_
            F=F_
            


            
        l=[]

        for j in range(d):
            l.append(np.linalg.norm(W[j,:],ord=2)+random.random()*1e-17)

        max_num_index_list = map(l.index, heapq.nlargest(round(rate*d), l))
        index_list=list(max_num_index_list)
        for each in index_list:
            each=each-1

        X_train=X_train[:,index_list]
        X_validation=X_validation[:,index_list]

        print(index_list)
        
        for i in range(F.shape[0]):
                for j in range(F.shape[1]):
                    if F[i,j]<=0.5:
                        F[i,j]=0
                    else:
                        F[i,j]=1

        Y_=copy.deepcopy(y_train)
        Y_[int(rate1*F.shape[0]):,:]=F[int(rate1*F.shape[0]):,:]
        

                            
        clf=BinaryRelevance(LinearSVC(random_state=0)).fit(X_train, Y_)
        score=clf.score(X_validation,y_validation)

        score_list.append(score)


    print(sum(score_list)/len(score_list))

    scores.append(sum(score_list)/len(score_list))
