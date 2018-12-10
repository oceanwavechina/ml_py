'''
Created on Dec 5, 2018

@author: liuyanan

@desc:
'''

import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn import svm
import warnings
warnings.filterwarnings(action="ignore", message="gelss")


def forcast_stock_closeprice():
    api_key = open('quandlapikey.txt', 'r').read().strip('\n')
    df = quandl.get('WIKI/GOOGL', authtoken=api_key)

    # volume指的是成交量
    df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

    # 最高价到最低价的变化
    df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100

    # 收盘价到开盘价的变化
    df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

    # 每一列就是一个feature
    df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

    forcast_col = 'Adj. Close'
    # 现实的世界中，删掉nan的数据会损失数据，所以尽量不要用dropna之类的方法
    df.fillna(-99999, inplace=True)

    # 这个意思是根据之前1/100的数据来预测
    forcast_out = int(math.ceil(0.01 * len(df)))
    print(forcast_out)

    # 这样 label中就是1/100后数据的'Adj. Close'和当天的 'Adj. Volume' 在一行了，
    # shift是负数是是往上移动, 下边的NAN补齐
    # label 列就是我们给的数据的标签，这些是历史的值，将来我们要预测的也是这列,
    # 所以这是监督学习
    df['label'] = df[forcast_col].shift(-forcast_out)
    df.dropna(inplace=True)  # 这行代码使得 X 和 y 的行数是一致的

    X = np.array(df.drop(['label'], 1))
    y = np.array(df['label'])

    # normalize
    X = preprocessing.scale(X)
    # X = X[:-forcast_out]  # 保证这些数据是有对应的 y 值得
    df.dropna(inplace=True)
    y = np.array(df['label'])
    print(len(X), len(y))   # 看看 X 和 y是不是一致的

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    clf = LinearRegression(n_jobs=-1)
    # clf = svm.SVR()
    # clf = svm.SVR(kernel='poly')
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(accuracy)


forcast_stock_closeprice()
