'''
Created on May 23, 2016

@author: liuyanan
'''

import numpy as np
from sklearn import linear_model, model_selection
import pandas as pd
import matplotlib.pyplot as plt

# 加载数据，指定X，y
ad_data = pd.read_csv("Advertising.csv", index_col=0)


def plot_correlation():
    # 看看销售额和哪个广告类型更相关
    df_corr = ad_data.corr()
    print(df_corr)
    print(np.array(df_corr['Sales']))
    col = df_corr['Sales'][:-1]
    lables = ['TV', 'Radio', 'Newspaper']
    plt.bar(np.arange(1, len(col) + 1, 1), np.array(col), tick_label=lables, width=0.4)
    plt.title('Sales Correlation')
    plt.show()


def plot_data():
    # 看看数据是什么样子的
    plt.plot(ad_data['Sales'], ad_data['TV'], 'ko', markersize=4, label='TV Cost')
    plt.plot(ad_data['Sales'], ad_data['Radio'], 'b^', markersize=4, label='Radio Cost')
    plt.plot(ad_data['Sales'], ad_data['Newspaper'], 'rx', markersize=4, label='Newspaper Cost')
    plt.xlabel('Sales')
    plt.ylabel('AD Cost')
    plt.legend(loc=2)
    plt.show()


X = np.array(ad_data[['TV', 'Radio', 'Newspaper']])
y = np.array(ad_data['Sales'])

#

# model_selection.train_test_split()
#
# # fit
# clf = linear_model.LinearRegression(normalize=True)
# # clf = linear_model.Ridge(alpha=.5)
# _data = np.c_[np.ones(len(data)), data[:, :3]]
# clf.fit(_data, data[:, 3:])
# print(clf.coef_)

if __name__ == '__main__':
    plot_data()
    plot_correlation()
