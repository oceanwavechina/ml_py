'''
Created on Dec 3, 2018

@author: liuyanan
'''
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def scatter():
    # 导入数据
    midwest = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/midwest_filter.csv")

    # 创建颜色，其实不用也行， matplotlib自己选的颜色就挺好
    categories = np.unique(midwest['category'])
    print(categories)
    colors = [plt.cm.tab10(i / float(len(categories) - 1)) for i in range(len(categories))]

    plt.figure(figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')

    #  enumerate相当于给元素加上index，这个挺有意思
    for i, category in enumerate(categories):
        # DataFrame.loc 是按label来获取元素, data中所有为给定category的数据
        plt.scatter('area', 'poptotal', data=midwest.loc[midwest.category == category, :], s=20, c=colors[i], label=str(category))
        # plt.scatter('area', 'poptotal', data=midwest.loc[midwest.category == category, :], s=20, label=str(category))
        # plt.plot('area', 'poptotal', data=midwest.loc[midwest.category == category, :], marker='o', linestyle='', label=str(category))

    plt.gca().set(xlim=(0.0, 0.1), ylim=(0, 90000), xlabel='Area', ylabel='Population')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Scatterplot of Midwest Area vs Population', fontsize=22)
    plt.legend(fontsize=12)

    plt.show()


if __name__ == '__main__':
    scatter()
