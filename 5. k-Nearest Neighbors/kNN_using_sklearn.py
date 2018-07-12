# -*- coding: utf-8 -*-
"""
Implementation of k-NN in Python using sklearn

@author: Batakrishna
Ref: 
[1] https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/ 
"""

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

data = pd.read_csv('iris.csv')
testSet = [[7.2, 3.6, 5.1, 2.5]]
test = pd.DataFrame(testSet)
k = 1

neigh = KNeighborsClassifier(n_neighbors=k)
neigh.fit(data.iloc[:,0:4], data['Name'])

print(neigh.predict(test))

print(neigh.kneighbors(test)[1])
