'''
Created on Dec 2, 2018

@author: liuyanan
'''

import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

api_key = open('quandlapikey.txt', 'r').read().strip('\n')
HPI_data = pd.read_pickle('pickle.pickle')


def add_column():
    print(HPI_data)
    HPI_data['AK_NSA_Value2'] = HPI_data['AK_NSA_Value'] * 2
    print(HPI_data[['AK_NSA_Value', 'AK_NSA_Value2']])


def grab_data_with_percentchange():
    fiddy_states = pd.read_html("https://simple.wikipedia.org/wiki/List_of_U.S._states")

    HPI_data = pd.DataFrame()
    for abbv in fiddy_states[0][1][1:6]:
        query = 'FMAC/HPI_' + str(abbv)
        df = quandl.get(query, authtoken=api_key)
        # 这里要让每个column的值唯一
        NSA_NAME = str(abbv) + '_NSA_Value'
        SA_NAME = str(abbv) + '_SA_Value'
        df.rename(columns={'NSA Value': NSA_NAME, 'SA Value': SA_NAME}, inplace=True)

        # percent change
        # df = df.pct_change()

        df[NSA_NAME] = (df[NSA_NAME] - df[NSA_NAME][0]) / df[NSA_NAME][0] * 100.0
        df[SA_NAME] = (df[SA_NAME] - df[SA_NAME][0]) / df[SA_NAME][0] * 100.0

        if HPI_data.empty:
            HPI_data = df
        else:
            HPI_data = HPI_data.join(df)

    print(HPI_data)

    HPI_data.to_pickle('pickle.pickle')

    HPI_data = pd.read_pickle('pickle.pickle')
    HPI_data.plot(linewidth=1)
    plt.legend().remove()
    plt.show()


def HPI_BenchMark():
    benchmarkdf = quandl.get('FMAC/HPI_USA', authtoken=api_key)
    print(benchmarkdf.head())

    COLNAME_1 = 'NSA Value'
    benchmarkdf[COLNAME_1] = (benchmarkdf[COLNAME_1] - benchmarkdf[COLNAME_1][0]) / benchmarkdf[COLNAME_1][0] * 100.0

    COLNAME_2 = 'SA Value'
    benchmarkdf[COLNAME_2] = (benchmarkdf[COLNAME_2] - benchmarkdf[COLNAME_2][0]) / benchmarkdf[COLNAME_2][0] * 100.0

    HPI_data = pd.read_pickle('pickle.pickle')

    fig = plt.figure()
    ax1 = plt.subplot2grid((1, 1), (0, 0))

    benchmarkdf.plot(ax=ax1, linewidth=10, color='k')
    HPI_data.plot(ax=ax1, linewidth=1)

    plt.legend().remove()
    plt.show()


def Correlation():

    HPI_data = pd.read_pickle('pickle.pickle')
    HPI_State_Correlation = HPI_data.corr()
    print(HPI_State_Correlation)

    print(HPI_State_Correlation.describe())


def resampling():
    fig = plt.figure()
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    HPI_data = pd.read_pickle('pickle.pickle')

    # AL1yr = HPI_data['AL_NSA_Value'].resample('A', how='mean')
    AL1yr = HPI_data['AL_NSA_Value'].resample('A', how='ohlc')
    print(AL1yr.head())

    HPI_data['AL_NSA_Value'].plot(ax=ax1, linewidth=1, label='Monthly AL HPI')
    AL1yr.plot(ax=ax1, label='Yearly AL HPI')

    plt.legend(loc=4)
    plt.show()


def handling_missing_data():
    fig = plt.figure()
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    HPI_data = pd.read_pickle('pickle.pickle')

    HPI_data['AL1yr'] = HPI_data['AL_NSA_Value'].resample('A', how='mean')
    print(HPI_data[['AL_NSA_Value', 'AL1yr']].head())
    # HPI_data.dropna(how='all', inplace=True)
    # HPI_data.fillna(method='ffill', inplace=True)
    # HPI_data.fillna(method='bfill', inplace=True)
    # HPI_data.fillna(method='bfill', inplace=True)

    # 这个相当于画一个阴影区域了
    HPI_data.fillna(value=-0, limit=10, inplace=True)
    print(HPI_data[['AL_NSA_Value', 'AL1yr']].head())

    # print(HPI_data.isnull().values.sum())

    HPI_data[['AL_NSA_Value', 'AL1yr']].plot(ax=ax1)

    plt.legend(loc=4)
    plt.show()


def roling_statistics():
    fig = plt.figure()
    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1)
    HPI_data = pd.read_pickle('pickle.pickle')
    HPI_data['AL12MA'] = HPI_data['AL_NSA_Value'].rolling(12).mean()
    HPI_data['AL12STD'] = HPI_data['AL_NSA_Value'].rolling(12).std()

    print(HPI_data[['AL_NSA_Value', 'AL12MA', 'AL12STD']].head())
    # HPI_data.dropna(inplace=True)

    HPI_data[['AL_NSA_Value', 'AL12MA']].plot(ax=ax1)
    HPI_data['AL12STD'].plot(ax=ax2)
    plt.legend(loc=4)
    plt.show()


def roling_correlation():
    fig = plt.figure()
    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1)

    HPI_data = pd.read_pickle('pickle.pickle')

    # 互相关性，如果为负数则两者在不同的方向增长
    AL_AK_12_corr = HPI_data['AL_NSA_Value'].rolling(12).corr(HPI_data['AK_NSA_Value'])

    HPI_data['AL_NSA_Value'].plot(ax=ax1, label='AL HPI')
    HPI_data['AK_NSA_Value'].plot(ax=ax1, label='AK HPI')
    ax1.legend(loc=4)

    AL_AK_12_corr.plot(ax=ax2, label='corr')

    plt.legend(loc=4)
    plt.show()


if __name__ == '__main__':
    # add_column()
    # grab_data_with_percentchange()
    # HPI_BenchMark()
    # Correlation()
    # resampling()
    # handling_missing_data()
    # roling_statistics()
    roling_correlation()
