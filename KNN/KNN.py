import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


basepath=os.path.abspath(os.path.dirname(__file__))
path1=basepath+'/semeion_train.csv'
path2=basepath+'/semeion_test.csv'

a=np.loadtxt(path1)
b=np.loadtxt(path2)

A=np.mat(a)
B=np.mat(b)
row1=len(A)
row2=len(B)


X1=A[:,0:256]
Y1=A[:,256:266]
X2=B[:,0:256]
Y2=B[:,256:266]



accuracy=[]
numbers1=[]
numbers2=[]

draw1=[]
draw2=[]


esp=0.00000000000000001




for k in range(1,25):
    
    correct=0
    for j in range(row2):#对验证集当中的每一个
        
        test=X2[j,:]

        distances=np.zeros((row1,1))

        for i in range(row1) :#跟训练集中的每一个求欧式距离
          z=X1[i,:]
          distances[i,0]=np.sqrt(np.sum(np.square(z - test)))
     
        sortedDistances=distances.copy()
        sortedDistances.sort(axis=0)

        index=np.zeros((k,1))

        for i in range(k):
            for t in range(row1):
                if t in index:
                    continue
                if(abs(sortedDistances[i,0]-distances[t,0])<esp):
                    index[i,0]=t
    
        similar=np.zeros((k,10))

        for i in range(k):
            t=int(index[i,0])
            similar[i,:]=Y1[t,:]

    
        count=np.zeros((10,1))

        for i in range(k):
          for t in range(10):
            if abs(similar[i,t]-1)<esp:
              count[t,0]=count[t,0]+1
          
   
        maxnum=max(count);

        number=0

        for i in range(10):
            if abs(maxnum[0]-count[i,0])<esp:
                number=i
    
        numbers2.append(number)
        
        if(abs(Y2[j,number]-1)<esp):
          correct=correct+1


        for t in range(10):
            if abs(Y2[j,t]-1)<esp:
                numbers1.append(t)


        testDraw=X2[j,:]
        for s in range(16):
            for t in range(16):
                if abs(testDraw[0,16*s+t]-1)<esp:
                    draw1.append(16-s)
                    draw2.append(t)
        

        """fig, ax = plt.subplots()
        ax.scatter(draw2,draw1,100)

        ax.set(xlabel='x', ylabel='y',title='test result')
       
        ax.grid(True)
        fig.tight_layout()
        fig.savefig("testDraw.png")
        plt.show()"""



    accuracy.append(correct/row2)



    

   

fig, ax = plt.subplots()
k=range(1,25)
ax.plot(k, accuracy)
ax.set(xlabel='K', ylabel='accuracy',
       title='KNN test result')
ax.grid()
fig.savefig("KNN.png")
plt.show()












