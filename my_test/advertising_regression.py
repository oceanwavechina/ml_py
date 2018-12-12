'''
Created on May 23, 2016

@author: liuyanan
'''

import numpy as np
from sklearn import linear_model, model_selection, preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker

# 加载数据，指定X，y
ad_data_all = pd.read_csv("Advertising.csv", index_col=0)
ad_data = ad_data_all[:-1]

fig = plt.figure(figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')


def plot_data():
    # 看看数据是什么样子的, 从图中可以看出，
    #     当费用较小时，报纸和广播的效果还是比较明显的，
    #     但是随着费用的增加，只有电视广告能呈现线性增长的态势
    ax1 = plt.subplot2grid((3, 1), (0, 0), fig=fig)
    ax1.plot(ad_data['Sales'], ad_data['TV'], 'ko', markersize=4, label='TV Cost')
    ax1.plot(ad_data['Sales'], ad_data['Radio'], 'b^', markersize=4, label='Radio Cost')
    ax1.plot(ad_data['Sales'], ad_data['Newspaper'], 'rx', markersize=4, label='Newspaper Cost')
    plt.title('AD_Type & Sales ralation')
    plt.xlabel('Sales')
    plt.ylabel('AD Cost')
    plt.legend(loc=2)


def plot_correlation():
    # 看看销售额和哪个广告类型更相关
    # 柱子越高则相关性越高
    ax2 = plt.subplot2grid((3, 1), (1, 0), colspan=1, fig=fig)
    df_corr = ad_data.corr()
    print(df_corr)
    print(np.array(df_corr['Sales']))
    col = df_corr['Sales'][:-1]
    lables = ['TV', 'Radio', 'Newspaper']
    ax2.bar(np.arange(1, len(col) + 1, 1), np.array(col), tick_label=lables, width=0.4)
    plt.title('Sales Correlation')
    plt.ylabel('value')
    plt.xlabel('Ad Types')


def sns_correlation():
    ax3 = plt.subplot2grid((3, 1), (2, 0), fig=fig)
    sns.heatmap(ad_data.corr(), xticklabels=ad_data.corr().columns, yticklabels=ad_data.corr().columns,
                cmap='RdYlGn', center=0, annot=True, ax=ax3)

    ax3.xaxis.set_major_locator(mticker.MaxNLocator(10))
    plt.title('AD correlation', fontsize=22)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)


def plot():
    plot_data()
    plot_correlation()
    sns_correlation()
    plt.tight_layout()  # 这个是使间距自动适应
    plt.show()


def trian_model():
    X = np.array(ad_data[['TV', 'Radio', 'Newspaper']])
    X = preprocessing.scale(X)
    y = np.array(ad_data['Sales'])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    clf = linear_model.LinearRegression()
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    print('accuracy: ', accuracy)

    # 预测一下，看看差多少
    forcast_sales = clf.predict([X[-1]])
    print('forcast_sales:', forcast_sales)
    print('real_sales:', ad_data[-1:]['Sales'])


if __name__ == '__main__':
    plot()
    trian_model()
