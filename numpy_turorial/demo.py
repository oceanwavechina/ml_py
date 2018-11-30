'''
Created on Nov 29, 2018

@author: liuyanan
'''

import numpy as np

L = [1, 2, 3, 4]
A = np.array(L)


def list_vs_array():
    """
    numpy 的 array 是按向量计算, 所以array的计算在用法上不用遍历
    python 的list是面向 元素的, 而且python的for-loop 相比之下效率是非常低的
    """
    print('2*L:', 2 * L)
    print('2*A:', 2 * A)
    print('\n')

    print('A**2:', A**2)
    print('\n')

    print('L+L', L + L)
    print('A+A:', A + A)
    print('\n')

    print('np.sqrt(A)', np.sqrt(A))
    print('np.log(A)', np.log(A))
    print('np.exp(A)', np.exp(A))
    print('\n')


def forloop_vs_cosinemethod_vs_dotfunction():
    a = np.array([1, 2])
    b = np.array([2, 1])

    dot = 0
    for e, f in zip(a, b):
        dot += e * f
    print('for-loop-dot', dot)

    print('np-dot', np.sum(a * b))
    print('np-dot', (a * b).sum())
    print('np-dot', a.dot(b))
    print('np-dot', b.dot(a))

    amag = np.sqrt((a * a).sum())
    print('amag', amag)

    amag = np.linalg.norm(a)
    cosangle = a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(cosangle)


def vector_and_matics():

    # array 和 matix 什么区别
    M = np.array([[1, 2], [3, 4]])
    print(M)
    print('M[0][0]', M[0][0])
    print('M[0,0]', M[0, 0])

    M2 = np.matrix([[1, 2], [3, 4]])
    print(M2)


def create_data():
    print(np.zeros(10))
    print(np.zeros((10, 10)))
    print(np.random.random((10, 10)))

    G = np.random.randn(10, 10)
    print(G)
    print('G.mean()', G.mean())
    print('G.var()', G.var())

    A = np.array([[1, 2], [3, 4]])
    Ainv = np.linalg.inv(A)
    print('A.dot(Ainv)', A.dot(Ainv))

    print('np.diag(A)', np.diag([1, 2, 3, 4, 5, 6, 7]))
    print('np.trace(A)', np.trace(A))

    a = np.array([1, 2])
    b = np.array([3, 4])

    print ('np.inner(a, b)', np.inner(a, b))


def linear_system():
    A = np.array([[1, 2], [3, 4]])
    b = np.array([1, 2])
    x = np.linalg.inv(A).dot(b)
    print('x', x)

    # 用solve不用要inv的方法
    x = np.linalg.solve(A, b)
    print('x', x)


def dummy():
    x = np.linspace(0, 10, 30, endpoint=False)
    print(dir(x), type(x), x)


if __name__ == '__main__':
    # list_vs_array()
    # forloop_vs_cosinemethod_vs_dotfunction()
    # dummy()
    # vector_and_matics()
    # create_data()
    linear_system()
