import numpy as np
import pandas as pd
import os
import copy
import math
import sklearn.neighbors.kde as kde
from sklearn.utils.testing import assert_allclose


basepath=os.path.abspath(os.path.dirname(__file__))
path=basepath+'/HWData3.txt'

a=np.loadtxt(path)
A=np.mat(a)


row=A.shape[0]
col=A.shape[1]
esp=0.000001
h=0.001

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

    
    pattern1=kde.KernelDensity(kernel="gaussian",bandwidth=h).fit(X1)
    pattern2=kde.KernelDensity(kernel="gaussian",bandwidth=h).fit(X2)
    pattern3=kde.KernelDensity(kernel="gaussian",bandwidth=h).fit(X3)

    log_dens11=pattern1.score_samples(test1)
    log_dens12=pattern2.score_samples(test1)
    log_dens13=pattern3.score_samples(test1)
    
    log_dens21=pattern1.score_samples(test2)
    log_dens22=pattern2.score_samples(test2)
    log_dens23=pattern3.score_samples(test2)

    log_dens31=pattern1.score_samples(test3)
    log_dens32=pattern2.score_samples(test3)
    log_dens33=pattern3.score_samples(test3)

    for i in range(10):
        if log_dens11[i]>=log_dens12[i] and log_dens11[i]>=log_dens13[i]:
            correct=correct+1

    for i in range(10):
        if log_dens22[i]>=log_dens21[i] and log_dens22[i]>=log_dens23[i]:
            correct=correct+1

    for i in range(10):
        if log_dens33[i]>=log_dens31[i] and log_dens33[i]>=log_dens32[i]:
            correct=correct+1
    





print(correct/150)


    
