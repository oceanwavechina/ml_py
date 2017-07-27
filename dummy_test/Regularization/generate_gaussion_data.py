'''
Created on May 23, 2016

@author: liuyanan
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# %matplotlib inline


def generate_1D_data(loc_positive=1, loc_negative=-1, size=100):
    X = np.hstack((np.random.normal(loc=loc_positive, size=size), np.random.normal(loc=loc_negative, size=size)))
    y = np.hstack((np.ones(size), np.zeros(size)))
    return X, y

X, y = generate_1D_data()
w = np.array([0, 0])


def prob(w, X):
    return 1 / (1 + np.exp(-w[0] - w[1] * X))


def cost(w, X, y, l2=0, l1=0, include_intercept=1):
    return -1 * np.sum(y * np.log(prob(w, X)) + (1 - y) * np.log(1 - prob(w, X))) + l2 * w[1] * w[1] + \
        include_intercept * l2 * w[0] * w[0] + l1 * np.abs(w[1]) + include_intercept * l1 * np.abs(w[0])


if __name__ == '__main__':
    #     plt.figure(figsize=(16, 8))
    #     plt.hist(np.random.normal(loc=1, size=100), alpha=0.5, color='steelblue', label='Positive Example')
    #     plt.hist(np.random.normal(loc=-1, size=100), alpha=0.5, color='darkred', label='Negative Example')
    #     plt.xlabel('Variable Value')
    #     plt.ylabel('Count')
    #     plt.title('Histogram of Simulated 1D Data')
    #     plt.show()

    delta = .05
    t1 = np.arange(-1.0, 1.0, delta)
    t2 = np.arange(0.0, 4.0, delta)
    T1, T2 = np.meshgrid(t1, t2)
    Z = T1.copy()
    for i, w0 in enumerate(t1):
        for j, w1 in enumerate(t2):
            Z[j, i] = cost([w0, w1], X, y)
    plt.figure(figsize=(16, 8))
    CS = plt.contour(T1, T2, Z, cmap=plt.cm.Blues_r)
    plt.clabel(CS, inline=1, fontsize=15, cmap=plt.cm.Blues)
    plt.xlabel('Intercept')
    plt.ylabel('First Coef')
    plt.title('Controur Plot of Negative Log Likelihood')
    plt.show()
