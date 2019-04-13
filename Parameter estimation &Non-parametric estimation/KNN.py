import numpy as np
import pandas as pd
import os
import copy
import math
import scipy


basepath=os.path.abspath(os.path.dirname(__file__))
path=basepath+'/HWData3.txt'

a=np.loadtxt(path)
A=np.mat(a)


row=A.shape[0]
col=A.shape[1]
esp=0.000001
k=5




def cal_probability(test,x1,x2,x3):
    
    distance1=np.zeros((1,40))
    for i in range(40):
        distance1[0,i]=np.linalg.norm(test-x1[i,:])
            
    distance2=np.zeros((1,40))
    for i in range(40):
        distance2[0,i]=np.linalg.norm(test-x2[i,:])
            
    distance3=np.zeros((1,40))
    for i in range(40):
        distance3[0,i]=np.linalg.norm(test-x3[i,:])

       
    sortedDistance1=distance1.copy()
    sortedDistance1.sort(axis=1)
        
    sortedDistance2=distance2.copy()
    sortedDistance2.sort(axis=1)

    sortedDistance3=distance3.copy()
    sortedDistance3.sort(axis=1)

    p1=0
    p2=0
    p3=0

    while 1:
        if(p1+p2+p3)>=k:
            break
        if sortedDistance1[0,p1]<=sortedDistance2[0,p2] and sortedDistance1[0,p1]<=sortedDistance3[0,p3]:
            p1=p1+1
            #print("p1++")
            continue
        if sortedDistance2[0,p2]<=sortedDistance1[0,p1] and sortedDistance2[0,p2]<=sortedDistance3[0,p3]:
            p2=p2+1
            #print("p2++")
            continue
        if sortedDistance3[0,p3]<=sortedDistance2[0,p2] and sortedDistance3[0,p3]<=sortedDistance1[0,p1]:
            p3=p3+1
            #print("p3++")
            continue

    prob1=p1/k
    prob2=p2/k
    prob3=p3/k

    return prob1,prob2,prob3




        
correct=0


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

   
   

    for test in test1:
        prob1,prob2,prob3=cal_probability(test,X1,X2,X3)
        print("%f %f %f"% (prob1,prob2,prob3))
        if prob1>=prob2 and prob1>=prob3:
            correct=correct+1
            
    for test in test2:
        prob1,prob2,prob3=cal_probability(test,X1,X2,X3)
        print("%f %f %f"% (prob1,prob2,prob3))
        if prob2>=prob1 and prob2>=prob3:
            correct=correct+1

    for test in test3:
        prob1,prob2,prob3=cal_probability(test,X1,X2,X3)
        print("%f %f %f"% (prob1,prob2,prob3))
        if prob3>=prob2 and prob3>=prob1:
            correct=correct+1

    

print("k=%d" % k)
print(correct/150)




