# -*- coding: utf-8 -*-
"""
Implementation of k-NN in Python from scratch

@author: Batakrishna
Ref: 
[1] https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/ 
"""

import pandas as pd
import numpy as np
import operator

data = pd.read_csv('iris.csv')

def euclideanDistance(data1, data2, length):
    distance = 0
    for x in range(length):
        distance += np.square(data1[x] - data2[x])
        
    return np.sqrt(distance) 

def knn(trainingSet, testInstance, k):
    distances = {}
    
    length = testInstance.shape[1]
    
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet.iloc[x], length)
        distances[x] = dist[0]
        
    sorted_d = sorted(distances.items(), key=operator.itemgetter(1))    
    
    neighbors = []
    
    for x in range(k):
        neighbors.append(sorted_d[x][0])
        
    classVotes = {}    

    for x in range(len(neighbors)):
        response = trainingSet.iloc[neighbors[x]][-1]
        
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1

    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    
    return(sortedVotes[0][0], neighbors)


    
testSet = [[7.2, 3.6, 5.1, 2.5]]
test = pd.DataFrame(testSet)
k = 1
result, neigh = knn(data, test, k)
print(result)
print(neigh)