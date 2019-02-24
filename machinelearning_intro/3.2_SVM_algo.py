'''
Created on Dec 22, 2018

@author: liuyanan

向量相关
    https://blog.csdn.net/dcrmg/article/details/52416832
    http://www.slimy.com/~steuard/teaching/tutorials/Lagrange.html
    https://www.zhihu.com/question/38586401/answer/134473412
    https://blog.csdn.net/ndzzl/article/details/79079561


预备知识：
    二向量点乘积几何意义：
        可以用来表征或计算两个向量的夹角，以及b向量在a向量上的投影
            a.b = |a||b|cosΘ  =>  Θ = acc cos( a.b / |a||b| )
        根据上述公式可以计算a，b两个向量的夹角，从而可以判断两个向量的方向关系
            a.b>0   同向，夹角在0到90度之间
            a.b=0   正交，垂直
            a.b<0   反向，夹角在90到180度之间
        

因为有
    w.x+b = 0  hyperplane (the decision boundary)
    w.x+b = 1  positive class(这个平面上的并不是所有的正分类的样本，而仅仅是support vector上的样本)
    w.x+b = -1  negative class(这个平面上的并不是所有的负分类的样本，而仅仅是support vector上的样本)
所以
    对于一个样本，其分类为 hyposisthy = sign(w.x+b)
所以
    我们要找到 hyposisthy 中的 w， b 是多少
其中
    the optimation objective is to minimize(||w||)(注意是w的norm) and maximize(b)
    the constraint is yi*(xi.w+b) >= 1 
        也就是 class * (knownFeatures . w + b) >= 1
        也就是，我们要找到满足上式的所有的 w, b 中最小的 w 和最大的 b
        w是一个shape为[5,3]的向量（假设我们有两个features）


这个naive的方法不是数学上推导的公式，而是根据svm的思想，用暴力破解的方法找到hyperplane
'''

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from numpy.__config__ import blas_opt_info

style.use('ggplot')


class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    def fit(self, data):        
        self.data = data
        # { ||w|| : [w, b] }
        opt_dict = {}

        '''
            我们在计算w的norm的时候，这些正负都不会对norm有影响，
            但是会影响w.x+b的方向
        '''
        transforms = [[1, 1], [-1, 1], [-1, -1], [1, -1]]

        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        # support vectors yi(xi.w+b) = 1
        #
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # points of expense:
                      self.max_feature_value * 0.001, ]

        # extramely expensive
        b_range_multiple = 5

        # we dont need to take as small of steps with b as we do w
        b_multiple = 5

        # first element of w
        latest_optimum = self.max_feature_value * 10

        '''
            这个方法是暴力破解
            这个for循环是不断测试越来越小的w
        '''
        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            # we can do this because convex
            optimized = False
            while not optimized:

                '''
                    这个for循环是找越来越大的b
                '''
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),
                                   self.max_feature_value * b_range_multiple,
                                   step * b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True
                        # weak link in the SVM fundamentally
                        # SMO attempts to fix this a bit
                        # yi(xi.w+b) >= 1
                        # ### add a break here later...
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    found_option = False
                                # print(xi, ':', yi * (np.dot(w_t, xi) + b))
                                # break
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                if w[0] < 0:
                    optimized = True
                    print('Optimized a step')
                else:
                    w = w - step

            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]

            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step * 2
        for i in self.data:
            for xi in self.data[i]:
                yi = i
                print(xi, ':', yi * (np.dot(self.w, xi) + self.b))

    def predict(self, features):
        # sign(x.w+b)
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 1 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])

        return classification

    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # hyperplane = w.x+b
        # v = x.w+b
        # psv=1, nsv=-1, dec_boundary=0
        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]

        datarange = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x+b)=1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

        # (w.x+b)=-1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        # (w.x+b)=0
        # decision boundray
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

        plt.show()


data_dict = {-1: np.array([[1, 7],
                           [2, 8],
                           [3, 8]]),
             1: np.array([[5, 1],
                          [6, -1],
                          [7, 3]]), }


svm = Support_Vector_Machine()
svm.fit(data_dict)

predict_us = [[0, 10],
              [1, 3],
              [3, 4],
              [3, 5],
              [5, 5],
              [5, 6],
              [6, -5],
              [5, 8],
              [1, 1]]

for p in predict_us:
    svm.predict(p)

svm.visualize()
