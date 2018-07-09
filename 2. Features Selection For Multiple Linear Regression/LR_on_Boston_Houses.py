# -*- coding: utf-8 -*-
"""
Following is a example of multiple regression in Boston Houses dataset.

@author: Batakrishna
"""
print(__doc__)

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

# load the dataset
boston = load_boston()

# describe the dataset
# print(boston.DESCR)

# load the features and label
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.DataFrame(boston.target, columns=['MEDV'])

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# define the classifier
clf = LinearRegression()

# fit the model
clf.fit(X_train, y_train)

# predict for testing set
y_pred = clf.predict(X_test)

# print the confidence score
print(clf.score(X_test, y_test))

# print the first 5 predicted values
print(y_pred[0:5])