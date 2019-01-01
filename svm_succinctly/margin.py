'''
Created on Dec 31, 2018

@author: liuyanan

margin 的计算方法

'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')


def example_functional_margin(w, b, x, y):
    '''
        1. 计算由 w 和 b 定义的 hyperplane(即 Y=wX+b) 上某个样本x的margin,反映的是该样本x和超平面有多远
            这个还不是到超平面的垂直距离
        2. 我么计算出来的值可能为正或为负, y是样本的label， wx+b是模型预测的记过，
            如果二者之积为正则是预测正确了
            如果为负则是预测错误了
    '''
    return y * (np.dot(w, x) + b)


def functional_margin(w, b, X, y):
    return np.min([example_functional_margin(w, b, x, y[i]) for i, x in enumerate(X)])


def example_geometric_margin(w, b, x, y):
    norm = np.linalg.norm(w)    # 向量w的length

    '''
        这里我们把定义平面(y=wx+b)的参数 w和b 都进行了单位化
        我们关心的是向量的方向
    '''
    return y * (np.dot(w / norm, x) + b / norm)


def geometric_margin(w, b, X, y):
    return np.min([example_geometric_margin(w, b, x, y[i]) for i, x in enumerate(X)])


def test_example():
    X = np.array([1, 1])
    y = 1

    b1 = 5
    w1 = np.array([2, 1])

    b2 = b1 * 10
    w2 = w1 * 10

    print('functional_margin(5, [2, 1]):', example_functional_margin(w1, b1, X, y))  # 8
    print('functional_margin(50, [20, 10]):', example_functional_margin(w2, b2, X, y))  # 80

    print('geometric_margin(5, [2, 1]):', example_geometric_margin(w1, b1, X, y))  # 8
    print('geometric_margin(50, [20, 10]):', example_geometric_margin(w2, b2, X, y))  # 8


def compare_two_hyperplane():
    '''
        样本数据，两个分类
    '''
    positive_x = [[2, 7], [8, 3], [7, 5], [4, 4], [4, 6], [1, 3], [2, 5]]
    negative_x = [[8, 7], [4, 10], [9, 7], [7, 10], [9, 6], [4, 8], [10, 10]]

    X = np.vstack((positive_x, negative_x))
    y = np.hstack((np.ones(len(positive_x)), -1 * np.ones(len(negative_x))))

    # 画出我们的样本
    plt.plot(np.reshape(X[:len(positive_x), 0], (-1,)), np.reshape(X[:len(positive_x), 1], (-1,)), 'o')
    plt.plot(np.reshape(X[len(positive_x):, 0], (-1,)), np.reshape(X[len(positive_x):, 1], (-1,)), '*')

    line_xs = np.array(range(0, 13))

    # 比较两个hyperplane的margin那个更大
    w = np.array([-0.4, -1])
    b = 8
    print('hyperplane of (w=[-0.4, -1], b=8): ', geometric_margin(w, b, X, y))
    line1_ys = [w[0] * xs + b for xs in line_xs]
    plt.plot(line_xs, line1_ys, color='b', label='hyperplane with b=8')

    b = 8.5
    print('hyperplane of (w=[-0.4, -1], b=8.5): ', geometric_margin(w, b, X, y))
    line1_ys = [w[0] * xs + b for xs in line_xs]
    plt.plot(line_xs, line1_ys, color='k', label='hyperplane with b=8.5')

    b = 9
    print('hyperplane of (w=[-0.4, -1], b=9): ', geometric_margin(w, b, X, y))
    line1_ys = [w[0] * xs + b for xs in line_xs]
    plt.plot(line_xs, line1_ys, color='g', label='hyperplane with b=9')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    # test_example()
    compare_two_hyperplane()
