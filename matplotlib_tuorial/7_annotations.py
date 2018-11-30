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
from matplotlib import style

style.use('ggplot')
# style.use('dark_background')
print(plt.style.available)


def bytespdata2num(fmt):
    strconverter = mdatas.strpdate2num(fmt)
    return strconverter


def annotation():
        # fig = plt.figure()
    ax1 = plt.subplot2grid((1, 1), (0, 0))

    date, closep, highp, lowp, openp, volume = np.loadtxt('601398.csv',
                                                          delimiter=',',
                                                          skiprows=1,
                                                          unpack=True,
                                                          encoding='utf-8',
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
    ax1.grid(True)

    # annotation example with arrow
    max_highp = np.max(highp)
    idx_max_highp = np.where(highp == max_highp)
    print(max_highp, idx_max_highp)
    ax1.annotate('Sell It!!', (date[idx_max_highp[0]], highp[idx_max_highp[0]]),
                 xytext=(1, 1), textcoords='axes fraction',
                 arrowprops=dict(facecolor='grey', color='grey'))

    min_lowp = np.min(lowp)
    idx_min_lowp = np.where(lowp == min_lowp)
    print(min_lowp, idx_min_lowp)
    ax1.annotate('Buy Now!', (date[idx_min_lowp[0][0]], lowp[idx_min_lowp[0][0]]),
                 xytext=(80, 120), textcoords='figure points',
                 arrowprops=dict(facecolor='grey', color='grey'))

    # font example
    font_dict = {'family': 'serif',
                 'color': 'red'}

    # text
    ax1.text(date[180], closep[120], 'ICBC price', fontdict=font_dict)

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('ICBC')
    # plt.legend()
    plt.subplots_adjust(left=0.09, bottom=0.16, right=0.94, top=0.90, wspace=0.9, hspace=0)
    plt.show()


def annotation_last():
        # fig = plt.figure()
    ax1 = plt.subplot2grid((1, 1), (0, 0))

    date, closep, highp, lowp, openp, volume = np.loadtxt('601398.csv',
                                                          delimiter=',',
                                                          skiprows=1,
                                                          unpack=True,
                                                          encoding='utf-8',
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
    ax1.grid(True)

    # annotation with style
    bbox_props = dict(boxstyle='larrow', fc='w', ec='k', lw=1)
    ax1.annotate(str(closep[-1]), (date[-1], closep[-1]),
                 xytext=(date[-1] + 20, closep[-1]), bbox=bbox_props)

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('ICBC')
    # plt.legend()
    plt.subplots_adjust(left=0.09, bottom=0.24, right=0.90, top=0.90, wspace=0.9, hspace=0)
    plt.show()


if __name__ == '__main__':
    # annotation()
    annotation_last()
