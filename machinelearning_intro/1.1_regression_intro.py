'''
Created on Dec 5, 2018

@author: liuyanan

NOTE:
    股票价格用线性回归会出现一个问题就是，之前如果是增长的态势，那以后都是增长的态势
    因为我们训练出来的曲线是增长的，所以在这些数据上虽然拟合很好，我们依然无法预测将来的走势
    这个程序中所选择的特征都不是影响google价值的特征，所以从现实意义上说，这样的预测没有价值
'''

import pandas as pd
import quandl
import datetime
import math
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn import svm
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')


def model_training():
    api_key = open('quandlapikey.txt', 'r').read().strip('\n')
    df = quandl.get('WIKI/GOOGL', authtoken=api_key)

    # volume指的是成交量
    df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

    # 最高价到最低价的变化
    df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100

    # 收盘价到开盘价的变化
    df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

    # 每一列就是一个feature, 这些特征中第一个是价格，其余都不是
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
    # df.dropna(inplace=True)  # 这行代码使得 X 和 y 的行数是一致的

    # X = np.array(df.drop(['label'], 1))
    X = np.array(df.drop(['label', 'Adj. Close'], 1))
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


def predict():
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
    forcast_out = int(math.ceil(0.1 * len(df)))
    print(forcast_out)

    # 这样 label中就是1/100后数据的'Adj. Close'和当天的 'Adj. Volume' 在一行了，
    # shift是负数是是往上移动, 下边的NAN补齐
    # label 列就是我们给的数据的标签，这些是历史的值，将来我们要预测的也是这列,
    # 所以这是监督学习
    df['label'] = df[forcast_col].shift(-forcast_out)

    X = np.array(df.drop(['label'], 1))
    # normalize
    X = preprocessing.scale(X)
    # 这个是我们要预测的参数
    X_lately = X[-forcast_out:]
    X = X[:-forcast_out]
    print(X_lately)

    df.dropna(inplace=True)
    y = np.array(df['label'])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    # 把训练的模型存储起来，再次使用的时候就不用重新训练了
    clf = LinearRegression(n_jobs=-1)
    # clf = svm.SVR()
    # clf = svm.SVR(kernel='poly')
    clf.fit(X_train, y_train)

    # save the classifier
    with open('linearregression.pickle', 'wb') as f:
        pickle.dump(clf, f)

    pickle_in = open('linearregression.pickle', 'rb')
    clf = pickle.load(pickle_in)

    accuracy = clf.score(X_test, y_test)
    forcast_set = clf.predict(X_lately)
    print(forcast_set, accuracy, forcast_out)
    df['Forcast'] = np.nan

    last_date = df.iloc[-1].name
    last_unix = last_date.timestamp()
    oneday = 86400
    next_unix = last_unix + oneday

    for i in forcast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += oneday
        # = 右边的第一个list是特征的个数
        df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

    print(df.tail())

    df['Adj. Close'].plot()
    # 这个forcast是我们预测的，和之前的数据画在一起
    df['Forcast'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()


# model_training()
predict()
