# -*- coding: utf-8 -*-
"""
Python implementation of Simple Linear Regression using statsmodels.
For explaination see the Jupyter Notebook.

@author: Batakrishna
"""
import warnings;warnings.filterwarnings('ignore')

import statsmodels.api as sm

dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]

# Input
X = [row[0] for row in dataset]

# Output
y = [row[1] for row in dataset]

# Add b_0
X = sm.add_constant(X)

# Fitting model
model = sm.OLS(y, X).fit()

# Prediction
predictions = model.predict(X) 

# Print out the statistics
print(model.summary())

