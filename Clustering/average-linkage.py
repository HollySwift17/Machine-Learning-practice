import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

def create_data(centers,num=100,std=0.7):
    '''
    生成用于聚类的数据集
    :param centers: 聚类的中心点组成的数组。如果中心点是二维的，则产生的每个样本都是二维的。
    :param num: 样本数
    :param std: 每个簇中样本的标准差
    :return: 用于聚类的数据集。是一个元组，第一个元素为样本集，第二个元素为样本集的真实簇分类标记
    '''
    X, labels_true = make_blobs(n_samples=num, centers=centers, cluster_std=std)
    return  X,labels_true

def plot_data(*data):
    '''
    绘制用于聚类的数据集
    :param data: 可变参数。它是一个元组。元组元素依次为：第一个元素为样本集，第二个元素为样本集的真实簇分类标记
    :return: None
    '''
    X,labels_true=data
    labels=np.unique(labels_true)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    colors='rgbyckm' # 每个簇的样本标记不同的颜色
    for i,label in enumerate(labels):
        position=labels_true==label
        ax.scatter(X[position,0],X[position,1],label="cluster %d"%label,
		color=colors[i%len(colors)])

    ax.legend(loc="best",framealpha=0.5)
    ax.set_xlabel("X[0]")
    ax.set_ylabel("Y[1]")
    ax.set_title("data")
    plt.show()


class AverageLinkage:

    def __init__(self, data, k):
        self.k = k
        self.data = data
        
        self.fit()

    def fit(self):
        n = len(self.data)
        self.clusters = {}

        for i in range(n):
          self.clusters[i] = []
          self.clusters[i].append(i)

        self.dist = np.sqrt((np.square(self.data[:,np.newaxis]-self.data).sum(axis=2)))
        self.dist_copy=self.dist.copy()

        
        for i in range(n-self.k):
            merge = self.merging()
            print(merge)
            self.clusters[merge[0]] = self.clusters[merge[0]] + self.clusters[merge[1]]
            self.clusters.pop(merge[1])
            for j in range(n):
                ssum=0
                for each in self.clusters[merge[0]]:
                    ssum=ssum+self.dist_copy[j,each]
                aver=ssum/len(self.clusters[merge[0]])
                self.dist[j,merge[0]]=aver
                self.dist[merge[0],j]=aver
            
                

        for i in range(self.k):
            while not i in self.clusters:
                for j in [x for x in list(map(int, self.clusters.keys())) if x >= i+1]:
                    self.clusters[j-1] = self.clusters.pop(j)

        for i in self.clusters.keys():
            self.clusters[i].sort()


    def merging(self):
        mini = 1e99 
        merge = (None, None)
        
        for i in list(map(int, self.clusters.keys())):
            for j in [x for x in list(map(int, self.clusters.keys())) if x >= i+1]:

                if self.dist[i][j] < mini:
                    mini = self.dist[i][j]
                    merge = (i, j)
                    
        return merge



if __name__=='__main__':
    centers=[[1,1,1],[1,3,3],[3,6,5],[2,6,8]] # 用于产生聚类的中心点, 聚类中心的维度代表产生样本的维度
    X,labels_true=create_data(centers,1000,0.5) # 产生用于聚类的数据集，聚类中心点的个数代表类别数
    print(X.shape)
    plot_data(X,labels_true)
    hc = SingleLinkage(X, 4)
    print(hc.clusters)
    
    labels_pred=[]
    row=X.shape[0]

    a=0
    b=0
    c=0
    d=0

    for i in range(row):
        if a<len(hc.clusters[0]) and hc.clusters[0][a]==i:
            labels_pred.append(0)
            a=a+1

        if b<len(hc.clusters[1]) and hc.clusters[1][b]==i:
            labels_pred.append(1)
            b=b+1

        if c<len(hc.clusters[2]) and hc.clusters[2][c]==i:
            labels_pred.append(2)
            c=c+1

        if d<len(hc.clusters[3]) and hc.clusters[3][d]==i:
            labels_pred.append(3)
            d=d+1


   
    plot_data(X,labels_pred)










            
    
