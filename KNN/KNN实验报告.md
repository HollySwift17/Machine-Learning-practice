# KNN手写数字识别实验报告

信息安全 李涵 1711290

### 一、问题描述

​	以semeion_train.csv作为训练集，semeion_test.csv为测试集，借助KNN算法，对于semeion_test中的每一组数据根据semeion_train进行预测，再与semeion_test中的结果进行比较，在不同的K值下统计准确率。

### 二、解决方法

1. 解决思路

   计算测试集与训练集中每个对象之间的距离，找出距离最小的k个，统计这k个中结果出现的最多的标签

   其他方法：线性回归

2. 基本理论

   (1)kNN算法：

   又称为k近邻分类(k-nearest neighbor classification)算法，属于机器学习中的监督学习，是指从训练集中找到和待分类数据最接近的k条记录，然后根据他们的标签来决定待分类数据的标签。该算法涉及3个主要因素：训练集、距离或相似的衡量、k的大小。

   交叉验证(Cross-Validation)：亦称循环估计，是一种统计学上将数据样本切割成较小子集的实用方法。可以先在一个子集上做分析， 而其它子集则用来做后续对此分析的确认及验证。 一开始的子集被称为训练集。而其它的子集则被称为验证集或测试集。

   

   (2) 线性回归原理：

   线性回归（Linear Regression）是一种通过属性的线性组合来进行预测的线性模型，其目的是找到一条直线或者一个平面或者更高维的超平面，使得预测值与真实值之间的误差最小化。

   公式：![clip_image005](http://images.cnblogs.com/cnblogs_com/jerrylead/201103/201103052209103916.png)为找到合适的θ使得h与标签接近

   在数据集过大的情况下（n>10^6）使用梯度下降来求解θ，在本例中，数据集仅为1500左右，所以可以直接而用矩阵计算来求取θ=![1551841487484](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1551841487484.png)

   再用θ和X相乘得到h，与Y比较，求取准确率

   

3. 算法流程

   KNN算法：

   1）对于每个给定测试对象，计算它与训练集中的每个对象的欧式距离

   2）按照距离的递增关系进行排序

   3）找到距离最近的k个训练对象，作为测试对象的近邻

   4）确定前K个点的标签的出现频率

   5）返回前K个点中出现频率最高的类别作为测试数据的预测分类。    



​	线性回归：

​	1）利用公式求取θ

​	2）h=X×θ 求取假设h

​	3）比较假设h和实际结果Y，求取精确度



###三、实验分析

1. 实验数据

    semeion中的每一组数据包含一个1×256的double型的0或1，即一个16×16个像素点的手写数字0-9，和一个1*10的向量，只有一个是1，哪个位置是1，就是哪个数字。

2. 实验设计（python语言实现）

    读取文件，并且将其转为矩阵

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

basepath=os.path.abspath(os.path.dirname(__file__))#本文件所在的地址
path1=basepath+'/semeion_train.csv'
path2=basepath+'/semeion_test.csv'

a=np.loadtxt(path1)
b=np.loadtxt(path2)#读取文件

A=np.mat(a)
B=np.mat(b)#转为矩阵
row1=len(A)
row2=len(B)#训练集和测试集的行数

X1=A[:,0:256]
Y1=A[:,256:266]
X2=B[:,0:256]
Y2=B[:,256:266]#把特征向量和标签（结果）向量分开

accuracy=[]#用来记录不同k值下的精确度
numbers1=[]
numbers2=[]

draw1=[]
draw2=[]#用作绘制散点图
esp=0.00000000000000001#精确度 用作double类的比较相等

```



​	对于每一个给定的k值（1~25），计算每一个验证集中的对象，与训练集之间的距离

```python
for j in range(row2):#对验证集当中的每一个
        
        test=X2[j,:]

        distances=np.zeros((row1,1))

        for i in range(row1) :#跟训练集中的每一个求欧式距离
          z=X1[i,:]
          distances[i,0]=np.sqrt(np.sum(np.square(z - test)))
```



​	对距离进行排序

```python
		sortedDistances=distances.copy()
        sortedDistances.sort(axis=0)
```



​	找到距离最小的k个



```python
    index=np.zeros((k,1))
    for i in range(k):
        for t in range(row1):
            if t in index:#如果t已在index中存在，则跳过继续，防止一个t被放在index里多次
                continue
            if(abs(sortedDistances[i,0]-distances[t,0])<esp):
                index[i,0]=t
```



​	找到最相近的k个的结果标签

```python
 		similar=np.zeros((k,10))

        for i in range(k):
            t=int(index[i,0])
            similar[i,:]=Y1[t,:]
```



​	统计k个近邻中是个标签出现的频率

```python
		count=np.zeros((10,1))

        for i in range(k):
          for t in range(10):
            if abs(similar[i,t]-1)<esp:
              count[t,0]=count[t,0]+1
```



​	找出频率最大的标签

```python
		maxnum=max(count);

        number=0

        for i in range(10):
            if abs(maxnum[0]-count[i,0])<esp:
                number=i
    
        numbers2.append(number)
        
         if(abs(Y2[j,number]-1)<esp):
         	correct=correct+1
```



​	绘制16×16的的散点图

```python
 		testDraw=X2[j,:]
        for s in range(16):
            for t in range(16):
                if abs(testDraw[0,16*s+t]-1)<esp:
                    draw1.append(16-s)
                    draw2.append(t)
        

        fig, ax = plt.subplots()
        ax.scatter(draw2,draw1,100)

        ax.set(xlabel='x', ylabel='y',title='test result')
       
        ax.grid(True)
        fig.tight_layout()
        fig.savefig("testDraw.png")
        plt.show()
```



​	统计各个k值的精确度并绘制折线图

```python
	accuracy.append(correct/row2)

fig, ax = plt.subplots()
k=range(1,25)
ax.plot(k, accuracy)
ax.set(xlabel='K', ylabel='accuracy',
       title='KNN test result')
ax.grid()
fig.savefig("KNN.png")
plt.show()

```



​	线性回归算法（matlab语言实现）

```matlab
load semeion.data;
semeion(1,:);

x=semeion(:,1:256);
y=semeion(:,257:266);

row=1593;
correct=0;

for i=1:row
%将每一个数据单独拿出来当验证集，其他作为训练集

X=ones(1593,257);
X(:,2:257)=x;
Y=y;

testX=X(i,:);
testY=Y(i,:);
%删除这一行

X(i,:)=[];
Y(i,:)=[];

theta=pinv(X'*X)*X'*Y;
%求取θ
h=testX*theta;
%求出假设h

    m1=0;
    for j=1:10
        if h(1,j)>m1
            m1=h(1,j);
            n1=j;
        end
      end

    m2=0;
    for j=1:10
        if testY(1,j)>m2
            m2=testY(1,j);
            n2=j;
        end
    end

	%比较h与实际结果
    if n1==n2
        correct=correct+1;
      end
      
end

accuracy=correct/row


```



3. 实验结果分析

（1）特定点的精确度：

​	K=1;	accuracy=0.85774 	错误率：0.14226

​	K=3;	accuracy=0.85146 	错误率：0.14854

​	K=5;	accuracy=0.87238 	错误率：0.12762

（2） 精确度随K值的变化：

![KNN](D:\Study\5\机器学习\1711290+李涵+第1次作业\相关绘图\KNN.png)

​	精确度随K值波动式下降，反映了KNN算法的K值并不是越大越好，反而在1~5时精确度较高，效果更好

![KNN2](D:\Study\5\机器学习\1711290+李涵+第1次作业\相关绘图\KNN2.png)

KNN2.py是导入semeion.txt，把1593个数据中的每一个单独拿出来，其他的1592个作为训练集，来推测拿出来的这一个的标签，跑起来可能比较慢，大概一个K值要跑10-15min，会比较慢

（3)绘制16×16的散点图，如图绘制的是一个2的散点图

![testDraw](D:\Study\5\机器学习\1711290+李涵+第1次作业\相关绘图\testDraw.png)

（4）线性回归算法

​	线性回归精确率0.84181，线性回归算法的运行时间较短

（5）KNN.py和KNN2.py的比较

​	KNN2的准确率要比KNN的准确率更高，说明随着训练集样本数的增加，KNN算法的精确度会提高

   (6)weka机器学习结果（k=1，10折交叉验证为例）

![1552377473511](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1552377473511.png)

注：需要对数据进行预处理，详情见traindata.arff



