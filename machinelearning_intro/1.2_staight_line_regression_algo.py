'''
Created on Dec 15, 2018

@author: liuyanan

https://www.youtube.com/watch?v=SvmueyhSkgQ&index=8&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v

NOTE:
    这是个通过均值计算给定样本的 拟合直线的方法

    R平方
        def R2(y_test, y_true):
            return 1 - ((y_test - y_true)**2).sum() / ((y_true - y_true.mean())**2).sum()

        1. R2方法是将预测值跟只使用均值的情况下相比，看能好多少

        2.  关于 (error^2) 为什么要取平方
        1). 因为我们关心的是 样本数据与拟合曲线 之间的的距离也就是偏离的程度，所以不关心方向
        2). 我们也可以取 4次方，6次方 等，取决于我们希望对于 outlier 的 惩罚程度
'''


from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)


def create_dataset(hm, variance, step=2, correlation=False):
    '''
    测试数据
        1. 我们可以根据测试数据的不同，看看是不是适合用线性回归
        2. 以及不同的数据分布和走向，模型的形状
    '''
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        # 因为y是纯随机的，所以我们通过加上一个值来决定这个在序列方向上是正相关还是负相关
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit_slop_and_intercept(xs, ys):
    # 根据样本计算模型
    m = ((mean(xs) * mean(ys) - mean(xs * ys)) / (mean(xs) * mean(xs) - mean(xs * xs)))
    b = mean(ys) - m * mean(xs)
    return m, b


def squared_error(ys_orig, ys_line):
    # return sum((ys_line - ys_orig) ** 2)
    # 因为这个差是在[0-1]之间，可以看到当取100次方的时候，这个平方和就为0了，也就是忽略了误差
    # 结果就是我们认为我们的预测数据100%和原有数据契合的
    #     print((ys_line - ys_orig) ** 100)
    #     print((ys_line - ys_orig) ** 2)
    return sum((ys_line - ys_orig) ** 2)


def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_mean = squared_error(ys_orig, y_mean_line)

    return 1 - (squared_error_regr / squared_error_mean)


# variance 越大，正确率越小
xs, ys = create_dataset(40, 40, 2, correlation='pos')
# variance 越小，正确率越大
xs, ys = create_dataset(40, 5, 2, correlation='neg')
xs, ys = create_dataset(40, 5, 2, correlation=False)

# 相当于fit的过程
m, b = best_fit_slop_and_intercept(xs, ys)
print(m, b)
# regression_line = [(m * x) + b for x in xs]
regression_line = m * xs + b    # numpy的计算直接是面向向量的，而且比python的for循环要快很多
# print(regression_line)

# 预测
predict_x = 8
predict_y = (m * predict_x) + b

# 我们可以通过这个值判断我们的数据适不适合用线性回归做拟合
r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)


def display():
    plt.scatter(xs, ys)
    plt.plot(xs, regression_line)
    plt.scatter(predict_x, predict_y, color='k', s=200)
    plt.show()


display()
