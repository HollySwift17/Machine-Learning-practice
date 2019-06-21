import scipy.io as scio
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
import numpy as np
import copy
from sklearn import preprocessing
import heapq
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.metrics import hamming_loss,log_loss
from sklearn.metrics import accuracy_score,recall_score,precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier,RidgeCV,LassoCV
import tensorflow as tf

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
b=np.ones((X.shape[0],1))
X=np.c_[X,b]


X_train,X_test,y_train,y_test= train_test_split(X,Y,test_size=0.2,random_state=1)
X_train,X_validation,y_train,y_validation= train_test_split(X_train,y_train,test_size=0.2,random_state=1)


n=X_train.shape[0]
d=X_train.shape[1]
c=y_train.shape[1]

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




#不同的损失函数


clf=BinaryRelevance(SGDClassifier(loss="hinge",penalty="l2")).fit(X_train,y_train)
pred=clf.predict(X_validation).toarray()

precision, recall=cal_precision_and_recall(y_validation,pred)
print(precision)
print(recall)

pause=input()


clf=BinaryRelevance(SGDClassifier(loss="log",penalty="l2")).fit(X_train,y_train)
pred=clf.predict(X_validation).toarray()
precision, recall=cal_precision_and_recall(y_validation,pred)
print(precision)
print(recall)

pause=input()



#不同的正则化项



clf=BinaryRelevance(RidgeCV()).fit(X_train,y_train)
pred=clf.predict(X_validation).toarray()
for i in range(pred.shape[0]):
    for j in range(pred.shape[1]):
        if pred[i,j]>=0.5:
            pred[i,j]=1
        else:
            pred[i,j]=0

precision, recall=cal_precision_and_recall(y_validation,pred)
print(precision)
print(recall)

pause=input()




clf=BinaryRelevance(LassoCV()).fit(X_train,y_train)
pred=clf.predict(X_validation).toarray()
for i in range(pred.shape[0]):
    for j in range(pred.shape[1]):
        if pred[i,j]>=0.5:
            pred[i,j]=1
        else:
            pred[i,j]=0

precision, recall=cal_precision_and_recall(y_validation,pred)
print(precision)
print(recall)

pause=input()




from keras import Sequential,regularizers
from keras.layers import Dense
from keras.models import Model
from keras import backend as K

model=Sequential()


def l1_reg(weight_matrix):
    W=K.batch_get_value(weight_matrix)
    D=np.eye(d)
    for j in range(d):
        D[j,j]=np.sum(W[j,i]*W[j,i]for i in range(c))
    return np.sum(D)



model.add(Dense(c, input_dim=d,kernel_regularizer=l1_reg))
model.compile(loss='hinge', optimizer='sgd')
for i in range(200):
    model.fit(X_train,y_train,batch_size=100,epochs=1)

pred=model.predict(X_validation)

for i in range(pred.shape[0]):
    for j in range(pred.shape[1]):
        if pred[i,j]>=0.5:
            pred[i,j]=1
        else:
            pred[i,j]=0

precision, recall=cal_precision_and_recall(y_validation,pred)
print(precision)
print(recall)










