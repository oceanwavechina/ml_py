'''
Created on Dec 1, 2018

@author: liuyanan
'''

import pandas as pd
import datetime
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')


def get_data_and_show():
    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime(2018, 1, 1)

    df = web.DataReader('XOM', "yahoo", start, end)

    print(df)
    df['Adj Close'].plot()

    plt.show()


def panda_basics():
    # 3个list必须是等长度的
    web_stats = {'Day': [1, 2, 3, 4, 5, 6],
                 'Visitors': [45, 67, 34, 78, 90, 44],
                 'Bounce_Rate': [65, 75, 58, 81, 66, 52]}

    df = pd.DataFrame(web_stats)

    print(df)
    print(df.head())
    print(df.tail(1))

    # 原来的dataframe不会变（另外list中的位置不会变）
    print(df.set_index('Day'))

    # 改变了原来的dataframe
    df.set_index('Day', inplace=True)
    print(df)
    print('\n')
    print(df['Visitors'])

    print(df[['Bounce_Rate', 'Visitors']])

    print(df.Visitors.tolist())

    print(np.array(df[['Bounce_Rate', 'Visitors']]).tolist())

    df2 = pd.DataFrame(np.array(df[['Bounce_Rate', 'Visitors']]))
    print(df2)


def io_basic():
    df = pd.read_csv('ZILLOW-Z77006_ZRISFRR.csv')
    print(df.head())
    print(df.tail())

    df.set_index('Date', inplace=True)
    df.to_csv('newcsv2.csv')

    # 在打开的时候指定index列
    df = pd.read_csv('ZILLOW-Z77006_ZRISFRR.csv', index_col=0)
    print(df.head())

    df.columns = ['house_price']
    print(df.head())
    df.to_csv('newcsv3.csv')
    df.to_csv('newcsv3_noheader.csv', header=False)

    df = pd.read_csv('newcsv3_noheader.csv', names=['Date', 'HousePrice'], index_col=0)
    print(df.head())

    df = pd.read_csv('newcsv3.csv')  # , names=['Date', 'HousePrice']) 加上这个参数会有两行header
    df.rename(index=str, columns={'HousePrice': '77006_price'}, inplace=True)
    print('\n')
    print(df.head())


if __name__ == '__main__':
    # get_data_and_show()
    # panda_basics()
    # io_basic()
    io_basic()
