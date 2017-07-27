'''
Created on May 23, 2016

@author: liuyanan
'''

import numpy as np
from sklearn import linear_model

# load data
tmp = np.loadtxt("Advertising.csv", np.str, delimiter=",")
data = tmp[1:, 1:].astype(np.float)
train
training_set = data[]

# fit
clf = linear_model.LinearRegression(normalize=True)
# clf = linear_model.Ridge(alpha=.5)
_data = np.c_[np.ones(len(data)), data[:, :3]]
print _data
clf.fit(_data, data[:, 3:])
print clf.coef_
