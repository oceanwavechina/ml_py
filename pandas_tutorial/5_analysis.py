'''
Created on Dec 2, 2018

@author: liuyanan
'''

import quandl
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from statistics import mean
from sklearn import svm, preprocessing, model_selection

style.use('fivethirtyeight')

api_key = open('quandlapikey.txt', 'r').read().strip('\n')


def dataframe_comparision():
    # 用来筛选数据
    bridge_height = {'meters': [10.26, 10.31, 10.27, 10.22, 10.23, 6212.42, 10.28, 10.25, 10.31]}
    df = pd.DataFrame(bridge_height)
    df['STD'] = df['meters'].rolling(2).std()
    print(df)

    df_std = df.describe()['meters']['std']
    print(df_std)

    df = df[(df['STD'] < df_std)]
    print(df)

    df['meters'].plot()
    plt.ylim((9, 11))
    plt.show()


def mortgage_30y():
    df = quandl.get('FMAC/MORTG', trim_start='1975-01-01', authtoken=api_key)

    #  相对于初始值的国债变化率
    COLNAME_1 = 'Value'
    df[COLNAME_1] = (df[COLNAME_1] - df[COLNAME_1][0]) / df[COLNAME_1][0] * 100.0
    df = df.resample('D').mean()
    df = df.resample('M').mean()
    df.columns = ['M30']
    return df


def HPI_BenchMark():
    benchmarkdf = quandl.get('FMAC/HPI_USA', authtoken=api_key)

    COLNAME_1 = 'NSA Value'
    benchmarkdf[COLNAME_1] = (benchmarkdf[COLNAME_1] - benchmarkdf[COLNAME_1][0]) / benchmarkdf[COLNAME_1][0] * 100.0

    COLNAME_2 = 'SA Value'
    benchmarkdf[COLNAME_2] = (benchmarkdf[COLNAME_2] - benchmarkdf[COLNAME_2][0]) / benchmarkdf[COLNAME_2][0] * 100.0

    benchmarkdf.rename(columns={COLNAME_1: 'BenchMark'}, inplace=True)
    return benchmarkdf['BenchMark']


def grab_initial_state_data():

    state_list = pd.read_html("https://simple.wikipedia.org/wiki/List_of_U.S._states")
    main_df = pd.DataFrame()

    for abbv in state_list[0][1][1:6]:
        query = 'FMAC/HPI_' + str(abbv)
        df = quandl.get(query, authtoken=api_key)

        # 这里要让每个column的值唯一, 因为join的时候要有不同的columnname
        df.rename(columns={'NSA Value': str(abbv)}, inplace=True)

        df = pd.DataFrame(df[str(abbv)], columns=[str(abbv), ])
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df)

    print(main_df.head())
    main_df.to_pickle('state.pickle')
    return main_df


def sp500_data():
    df = quandl.get('MULTPL/SP500_DIV_MONTH', trim_start='1975-01-01', authtoken=api_key)
    COLNAME_1 = 'Value'
    df[COLNAME_1] = (df[COLNAME_1] - df[COLNAME_1][0]) / df[COLNAME_1][0] * 100.0
    df = df.resample('M').mean()
    df.rename(columns={COLNAME_1: 'sp500'}, inplace=True)
    df = df['sp500']
    return df


def gdp_data():
    df = quandl.get('BCB/4385', trim_start='1975-01-01', authtoken=api_key)

    COLNAME_1 = 'Value'
    df[COLNAME_1] = (df[COLNAME_1] - df[COLNAME_1][0]) / df[COLNAME_1][0] * 100.0
    df = df.resample('M').mean()
    df.rename(columns={COLNAME_1: 'GDP'}, inplace=True)
    df = df['GDP']
    return df


def us_unemployment():
    df = quandl.get('USMISERY/INDEX', trim_start='1975-01-01', authtoken=api_key)

    COLNAME_1 = 'Unemployment Rate'
    df[COLNAME_1] = (df[COLNAME_1] - df[COLNAME_1][0]) / df[COLNAME_1][0] * 100.0
    df = df.resample('1D').mean()
    df = df.resample('M').mean()
    return df[COLNAME_1]


def joining_more_data():

    # grab_initial_state_data()
    HPI_data = pd.read_pickle('state.pickle')

    # 特征
    HPI_Beanch = HPI_BenchMark()
    m30 = mortgage_30y()
    sp500 = sp500_data()
    US_GDP = gdp_data()
    US_unemployemt = us_unemployment()

    HPI = HPI_data.join([HPI_Beanch, m30, sp500, US_GDP, US_unemployemt])
    HPI.dropna(inplace=True)
    HPI.to_pickle('HPI.pickle')
    print(HPI.head())

    # 这个是计算 房价和国债利率的相关性
    print(HPI.corr()['M30'].describe())


def create_labels(cur_hpi, fut_hpi):
    if fut_hpi > cur_hpi:
        return 1
    else:
        return 0


def moving_average(values):
    return mean(values)


def rolling_and_mapping():
    housing_data = pd.read_pickle('HPI.pickle')
    housing_data = housing_data.pct_change()

    housing_data.replace([np.inf, -np.inf], np.nan, inplace=True)

    housing_data['US_HPI_future'] = housing_data['BenchMark'].shift(-1)

    # print(housing_data[['US_HPI_future', 'BenchMark']].head())
    housing_data['label'] = list(map(create_labels, housing_data['BenchMark'], housing_data['US_HPI_future']))

    housing_data['ma_apply_example'] = housing_data['M30'].rolling(10).apply(moving_average, raw=True)
    housing_data.dropna(inplace=True)
    return housing_data


def machine_learning():
    housing_data = rolling_and_mapping()
    X = np.array(housing_data.drop(['label', 'US_HPI_future'], 1))
    X = preprocessing.scale(X)
    y = np.array(housing_data['label'])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    # 根据数据训练一个模型
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)

    print(clf.score(X_test, y_test))
    # 训练出来参数
    print(clf.coef_)
    # print(housing_data.drop(['label', 'US_HPI_future'], 1))


if __name__ == '__main__':
    #  dataframe_comparision()
    # joining_more_data()
    # rolling_and_mapping()
    machine_learning()
