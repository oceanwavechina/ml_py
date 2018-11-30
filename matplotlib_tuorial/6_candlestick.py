# coding:utf-8
'''
Created on Nov 18, 2018

@author: liuyanan

https://www.youtube.com/watch?v=ZyTO4SwhSeE&list=PLQVvvaa0QuDfefDfXb9Yf0la1fPDKluPF&index=3
'''

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdatas
import matplotlib.ticker as mticker
from mpl_finance import candlestick_ohlc as candlestick_ohlc


def bytespdata2num(fmt):
    strconverter = mdatas.strpdate2num(fmt)
    return strconverter


def spines_and_horizontal_line():
        # fig = plt.figure()
    ax1 = plt.subplot2grid((1, 1), (0, 0))

    date, closep, highp, lowp, openp, volume = np.loadtxt('601398.csv',
                                                          delimiter=',',
                                                          skiprows=1,
                                                          unpack=True,
                                                          encoding='utf8',
                                                          converters={0: bytespdata2num('%Y-%m-%d')},
                                                          usecols=(0, 3, 4, 5, 6, 13))

    x = 0
    y = len(date)
    ohlc = []

    while x < y:
        append_me = date[x], openp[x], highp[x], lowp[x], closep[x], volume[x]
        ohlc.append(append_me)
        x += 1

    candlestick_ohlc(ax1, ohlc, width=0.6, colorup='r', colordown='g')

    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)
    ax1.xaxis.set_major_formatter(mdatas.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('ICBC')
    plt.legend()
    plt.subplots_adjust(left=0.09, bottom=0.16, right=0.94, top=0.90, wspace=0.9, hspace=0)
    plt.show()


if __name__ == '__main__':
    # basic_customization()
    spines_and_horizontal_line()
