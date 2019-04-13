import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


basepath=os.path.abspath(os.path.dirname(__file__))
path=basepath+'/semeion.txt'

a=np.loadtxt(path)

A=np.mat(a)
row=1593
correct=0

X=A[:,0:256]
Y=A[:,256:266]
accuracy=[]
numbers1=[]
numbers2=[]

draw1=[]
draw2=[]


esp=0.00000000000000001




for k in range(1,25):
    
    correct=0
    for j in range(row):
        
        test=X[j,:]

        distances=np.zeros((row,1))


        for i in range(row) :
          z=X[i,:]
          distances[i,0]=np.sqrt(np.sum(np.square(z - test)))
     
        sortedDistances=distances.copy()
        sortedDistances.sort(axis=0)

        index=np.zeros((k,1))

        for i in range(k):
            for t in range(row):
                if t==j:
                    continue
                if t in index:
                    continue
                if(abs(sortedDistances[i+1,0]-distances[t,0])<esp):
                    index[i,0]=t
    
        similar=np.zeros((k,10))

        for i in range(k):
            t=int(index[i,0])
            similar[i,:]=Y[t,:]

    
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
        
        if(abs(Y[j,number]-1)<esp):
          correct=correct+1


        for t in range(10):
            if abs(Y[j,t]-1)<esp:
                numbers1.append(t)


        testDraw=X[j,:]
        for s in range(16):
            for t in range(16):
                if abs(testDraw[0,16*s+t]-1)<esp:
                    draw1.append(16-s)
                    draw2.append(t)
        

        """fig, ax = plt.subplots()
        ax.scatter(draw2,draw1,1000)

        ax.set(xlabel='x', ylabel='y',title='test result')
       
        ax.grid(True)
        fig.tight_layout()
        fig.savefig("testDraw.png")
        plt.show()
      """

     

    accuracy.append(correct/row)

   

fig, ax = plt.subplots()
k=range(1,25)
ax.plot(k, accuracy)
ax.set(xlabel='K', ylabel='accuracy',
       title='KNN test result')
ax.grid()
fig.savefig("KNN2.png")
plt.show()












