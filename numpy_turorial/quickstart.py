'''
Created on Dec 3, 2018

@author: liuyanan
'''

import numpy as np
from numpy import pi


def exercise_1():
    a = np.arange(15).reshape(3, 5)
    print('np.arange(15): ', np.arange(15))
    print('a:', a)
    print('a.shape:', a.shape)
    print('a.ndim:', a.ndim)
    print('a.dtype.name:', a.dtype.name)

    a = np.array([1, 2, 'a'])
    print(a)

    a = np.zeros((3, 4))
    print(a)

    a = np.linspace(0, 2 * pi, 99)
    print(a)
    print('pi:', pi)

    a = np.arange(15).reshape(3, 5)
    print('a:', a)
    print('a.sum:', a.sum())

    # axis=0 按行相加
    print('a.sum(axis=0):', a.sum(axis=0))
    print('a.sum(axis=1):', a.sum(axis=1))
    print('a.sum(axis=-1):', a.sum(axis=-1))

    print('a.all():', a.all())


if __name__ == '__main__':
    exercise_1()
