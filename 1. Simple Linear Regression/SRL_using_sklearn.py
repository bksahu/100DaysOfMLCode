# -*- coding: utf-8 -*-
"""
Python implementation of Simple Linear Regression using sklearn.
For explaination see the Jupyter Notebook.

@author: Batakrishna
"""
import warnings;warnings.filterwarnings('ignore')

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]

# Input
X = np.asarray([row[0] for row in dataset]).reshape(-1, 1)

# Output
y = np.asarray([row[1] for row in dataset]).reshape(-1, 1)

# define classifier
regr = linear_model.LinearRegression()

# Fit the model
regr.fit(X, y)

# Prediction
predictions = regr.predict(X) 

# The coefficients
print('Coefficients: \n', regr.coef_)

# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y, predictions))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y, predictions))