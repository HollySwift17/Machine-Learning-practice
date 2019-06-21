#!/usr/bin/python
# -*- coding: UTF-8 -*-


 

from numpy import *
import numpy as np
import pandas as pd
from math import log
import operator
import copy



def cal_gini(dataSet):
    numEntries=len(dataSet)
    labelCounts={}
    #给所有可能分类创建字典
    for featVec in dataSet:
        currentLabel=featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    gini=1.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        gini-=prob*prob
    return gini

 

#对离散变量划分数据集，取出该特征取值为value的所有样本

def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

 
#连续的取值采用分裂点进行分裂
def splitContinuousDataSet(dataSet,axis,value,direction):
    retDataSet=[]
    for featVec in dataSet:
        if direction==0:
            if featVec[axis]>value:
                reducedFeatVec=featVec[:axis]
                reducedFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reducedFeatVec)
        else:
            if featVec[axis]<=value:
                reducedFeatVec=featVec[:axis]
                reducedFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reducedFeatVec)
    return retDataSet

 

#选择最好的数据集划分方式

def chooseBestFeatureToSplit(dataSet,labels):
    numFeatures=len(dataSet[0])-1
    baseEntropy=cal_gini(dataSet)
    bestInfoGain=0.0
    bestFeature=-1
    bestSplitDict={}
    for i in range(numFeatures):
        featList=[example[i] for example in dataSet]
        #对连续型特征进行处理
        if type(featList[0]).__name__=='float' or type(featList[0]).__name__=='int':
            #产生n-1个候选划分点
            sortfeatList=sorted(featList)
            splitList=[]
            for j in range(len(sortfeatList)-1):
                splitList.append((sortfeatList[j]+sortfeatList[j+1])/2.0)
            

            bestSplitEntropy=10000
            slen=len(splitList)
            
            #求用第j个候选划分点划分时，得到的信息熵，并记录最佳划分点
            for j in range(slen):
                value=splitList[j]
                newEntropy=0.0
                subDataSet0=splitContinuousDataSet(dataSet,i,value,0)
                subDataSet1=splitContinuousDataSet(dataSet,i,value,1)
                prob0=len(subDataSet0)/float(len(dataSet))
                newEntropy+=prob0*cal_gini(subDataSet0)
                prob1=len(subDataSet1)/float(len(dataSet))
                newEntropy+=prob1*cal_gini(subDataSet1)
                if newEntropy<bestSplitEntropy:
                    bestSplitEntropy=newEntropy
                    bestSplit=j


            #用字典记录当前特征的最佳划分点

            bestSplitDict[labels[i]]=splitList[bestSplit]
            infoGain=baseEntropy-bestSplitEntropy

        #对离散型特征进行处理

        else:
            uniqueVals=set(featList)
            newEntropy=0.0
            #计算该特征下每种划分的信息熵
            for value in uniqueVals:
                subDataSet=splitDataSet(dataSet,i,value)
                prob=len(subDataSet)/float(len(dataSet))
                newEntropy+=prob*cal_gini(subDataSet)
                
            infoGain=baseEntropy-newEntropy
        if infoGain>bestInfoGain:
            bestInfoGain=infoGain
            bestFeature=i

    #若当前节点的最佳划分特征为连续特征，则将其以之前记录的划分点为界进行二值化处理

    #即是否小于等于bestSplitValue

    if type(dataSet[0][bestFeature]).__name__=='float' or type(dataSet[0][bestFeature]).__name__=='int':      
        bestSplitValue=bestSplitDict[labels[bestFeature]]        
        labels[bestFeature]=labels[bestFeature]+'<='+str(bestSplitValue)
        for i in range(shape(dataSet)[0]):
            if dataSet[i][bestFeature]<=bestSplitValue:
                dataSet[i][bestFeature]=1
            else:
                dataSet[i][bestFeature]=0

    return bestFeature

 

#特征若已经划分完，节点下的样本还没有统一取值，则需要进行投票

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    return max(classCount)

 

#主程序，递归产生决策树
def createTree(dataSet,labels,data_full,labels_full):
    classList=[example[-1] for example in dataSet]
    if classList.count(classList[0])==len(classList):
        return classList[0]

    if len(dataSet[0])==1:
        return majorityCnt(classList)

    bestFeat=chooseBestFeatureToSplit(dataSet,labels)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)

    if type(dataSet[0][bestFeat]).__name__=='str':
        currentlabel=labels_full.index(labels[bestFeat])
        featValuesFull=[example[currentlabel] for example in data_full]
        uniqueValsFull=set(featValuesFull)

    del(labels[bestFeat])

    #针对bestFeat的每个取值，划分出一个子树。
    for value in uniqueVals:
        subLabels=labels[:]
        if type(dataSet[0][bestFeat]).__name__=='str':
            uniqueValsFull.remove(value)
        myTree[bestFeatLabel][value]=createTree(splitDataSet\
         (dataSet,bestFeat,value),subLabels,data_full,labels_full)

    if type(dataSet[0][bestFeat]).__name__=='str':
        for value in uniqueValsFull:
            myTree[bestFeatLabel][value]=majorityCnt(classList)

    return myTree









df=pd.read_csv('Watermelon-train2.csv',encoding='ANSI')
data=df.values[:,1:].tolist()
data_full=data[:]
labels=df.columns.values[1:-1].tolist()
labels_full=labels[:]
myTree=createTree(data,labels,data_full,labels_full)


df1=pd.read_csv('Watermelon-test2.csv',encoding='ANSI')
data1=df1.values[:,1:6].tolist()
reality=df1.values[:,6].tolist()
length=len(data1)
correct=0
i=0

for each in data1:
    tree=copy.copy(myTree)
    deep=0
    while 1:
        
        curr_feature=list(tree.keys())[0]
        all_feature_labels=list(list(tree.values())[0].keys())
        temp=list(list(tree.values())[0].values())
        p=0

        if '<=' in curr_feature:#连续型变量
            l=curr_feature.split('<=')
            num=float(l[1])
            if each[4]>num:
                tree=temp[1]
            else:
                tree=temp[0]
                

        else:
            while 1:
                if all_feature_labels[p] in each:
                    break
                p=p+1
            
            tree=temp[p]

        
        if '是' == tree or '否' ==tree:
            print(tree)
            if tree==reality[i]:
                correct=correct+1

            i=i+1
            break
            

print(correct/len(data1))






