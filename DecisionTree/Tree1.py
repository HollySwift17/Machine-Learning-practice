#!/usr/bin/python
# -*- coding: UTF-8 -*-


 

from numpy import *
import numpy as np
import pandas as pd
from math import log
import operator
import copy


def calcShannonEnt(dataSet):
    numEntries=len(dataSet)
    labelCounts={}
    for featVec in dataSet:
        currentLabel=featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    shannonEnt=0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt-=prob*log(prob,2)
    return shannonEnt

 



def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

 

def chooseBestFeatureToSplit(dataSet,labels):
    numFeatures=len(dataSet[0])-1
    baseEntropy=calcShannonEnt(dataSet)
    bestInfoGain=0.0
    bestFeature=-1
    bestSplitDict={}
    for i in range(numFeatures):
        featList=[example[i] for example in dataSet]
        
        uniqueVals=set(featList)
        newEntropy=0.0
        #计算该特征下每种划分的信息熵
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calcShannonEnt(subDataSet)
                
        infoGain=baseEntropy-newEntropy
        if infoGain>bestInfoGain:
            bestInfoGain=infoGain
            bestFeature=i

   
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





df=pd.read_csv('Watermelon-train1.csv',encoding='ANSI')
data=df.values[:,1:].tolist()
data_full=data[:]
labels=df.columns.values[1:-1].tolist()
labels_full=labels[:]
myTree=createTree(data,labels,data_full,labels_full)



df1=pd.read_csv('Watermelon-test1.csv',encoding='ANSI')
data1=df1.values[:,1:5].tolist()
reality=df1.values[:,5].tolist()
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






