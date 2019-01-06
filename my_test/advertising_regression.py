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

'''
NOTE:
1.  适用 线性回归 的数据，一般我们都能从数据的分布中看出来，
    所以先把原始数据测分布图呈现出来，会对模型的选取比较有帮助
    因为对于数据分布不呈线性的情况，适用线性回归，结果不会太好
'''

# 加载数据，指定X，y
ad_data = pd.read_csv("Advertising.csv", index_col=0)
ad_data['forcast'] = np.nan


fig = plt.figure(figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
# 原始数据分布
ax1 = plt.subplot2grid((3, 1), (0, 0), fig=fig)


def plot_data():
    # 看看数据是什么样子的, 从图中可以看出，
    #     当费用较小时，报纸和广播的效果还是比较明显的，
    #     但是随着费用的增加，只有电视广告能呈现线性增长的态势
    # 因为这个是3个变量的数据，在二维的图中不能呈现所以只要每个变量单独画一下
    ax1.plot(ad_data['TV'], ad_data['Sales'], 'ko', markersize=4, label='TV Cost')
    ax1.plot(ad_data['Radio'], ad_data['Sales'], 'b^', markersize=4, label='Radio Cost')
    ax1.plot(ad_data['Newspaper'], ad_data['Sales'], 'rx', markersize=4, label='Newspaper Cost')
    plt.title('AD_Type & Sales ralation')
    plt.xlabel('AD Cost')
    plt.ylabel('Sales')
    plt.legend(loc=2)


def plot_correlation():
    # 看看销售额和哪个广告类型更相关
    # 柱子越高则相关性越高
    ax2 = plt.subplot2grid((3, 1), (1, 0), colspan=1, fig=fig)
    df_corr = ad_data.corr()
    print('corr:\n', df_corr)
    col = df_corr['Sales'].drop(['forcast', 'Sales'])
    lables = ['TV', 'Radio', 'Newspaper']
    ax2.bar(np.arange(1, len(col) + 1, 1), np.array(col), tick_label=lables, width=0.4)
    plt.title('Sales Correlation')
    plt.ylabel('value')
    plt.xlabel('Ad Types')


def sns_correlation():
    ax3 = plt.subplot2grid((3, 1), (2, 0), fig=fig)
    corr_data = ad_data[['TV', 'Radio', 'Newspaper']]
    sns.heatmap(corr_data.corr(), xticklabels=corr_data.corr().columns, yticklabels=corr_data.corr().columns,
                cmap='RdYlGn', center=0, annot=True, ax=ax3)

    ax3.xaxis.set_major_locator(mticker.MaxNLocator(10))
    plt.title('AD correlation', fontsize=22)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)


def plot():
    plot_data()
    plot_correlation()
    sns_correlation()
    plt.tight_layout(pad=1)  # 这个是使间距自动适应
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
    print('coef:', clf.coef_)
    print('param:', clf.get_params())

    # 预测 1 条数据，看看差多少
    forcast_sales = clf.predict([X[-1]])
    print('forcast_sales:', forcast_sales)
    print('real_sales:', ad_data[-1:]['Sales'])

    '''
    学习出来的曲线咋画？
        好像没法画，因为是多变量的

    # plot
    ad_data['forcast'] = clf.predict(X)
    print('data:\n', ad_data.head())

    ax1.plot(ad_data['TV'], ad_data['forcast'], 'yo', markersize=4)

#     radio_range = range(int(ad_data['Radio'].min()), int(ad_data['Radio'].max()), int(ad_data['Radio'].ptp()))
#     ax1.plot(radio_range, forcast_sales, 'b-')

#     newspaper_range = range(int(ad_data['Newspaper'].min()), int(ad_data['Newspaper'].max()), int(ad_data['Newspaper'].ptp()))
#     ax1.plot(newspaper_range, forcast_sales[:len(newspaper_range)], 'r-', markersize=4)
    '''


if __name__ == '__main__':
    plot()
    trian_model()
