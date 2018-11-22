# coding:utf-8
'''
Created on Nov 18, 2018

@author: liuyanan

https://www.youtube.com/watch?v=ZyTO4SwhSeE&list=PLQVvvaa0QuDfefDfXb9Yf0la1fPDKluPF&index=3
'''

import matplotlib.pyplot as plt


def line():
    x = [1, 2, 3]
    y = [5, 7, 4]

    x2 = [1, 2, 3]
    y2 = [10, 14, 12]

    plt.plot(x, y, label='First Line')
    plt.plot(x2, y2, label='Second Line')
    plt.xlabel('Plot Number')
    plt.ylabel('Important var')
    plt.title('Interesting Graph\nCheck it out')

    # 添加图例
    plt.legend()
    plt.show()


def barchart():
    x = [2, 4, 6, 8, 10]
    y = [6, 7, 8, 2, 4]

    x2 = [1, 3, 5, 9, 11]
    y2 = [7, 3, 9, 12, 5]

    plt.bar(x, y, label='bars1', color='r')
    plt.bar(x2, y2, label='bars2', color='c')

    plt.xlabel('Plot Number')
    plt.ylabel('Important var')
    plt.title('Interesting Graph\nCheck it out')
    plt.legend()

    plt.show()


def histograms():
    # 这个不是简单的罗列，而是按给定的区间统计各个区间的的个数，可以看到数据的分布
    population_ages = [22, 34, 45, 76, 73, 85, 98, 2, 39, 120, 11, 77, 98, 33, 54, 89, 83, 27, 29, 10, 30, 70]

#     ids = [x for x in range(len(population_ages))]
#     plt.bar(ids, population_ages)

    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 130]

    plt.hist(population_ages, bins, histtype='bar', rwidth=0.8)

    plt.xlabel('Plot Number')
    plt.ylabel('Important var')
    plt.title('Interesting Graph\nCheck it out')
    plt.legend()

    plt.show()


def scatter_plots():
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    y = [4, 6, 7, 1, 9, 2, 8, 2]

    plt.scatter(x, y, label='skitcat')

    plt.xlabel('Plot Number')
    plt.ylabel('Important var')
    plt.title('Interesting Graph\nCheck it out')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # line()
    # barchart()
    # histograms()
    scatter_plots
