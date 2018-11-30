'''
Created on Nov 28, 2018

@author: liuyanan
'''

import numpy as np
import matplotlib.pyplot as plt


def simple():

    # 准备 x轴 和 y轴 的数据
    X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    C, S = np.cos(X), np.sin(X)
    data = zip(X, C, S)
    [print(item) for item in data]

    plt.plot(X, C)
    plt.plot(X, S)

    plt.show()


if __name__ == '__main__':
    simple()
    pass
