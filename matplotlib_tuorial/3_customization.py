# coding:utf-8
'''
Created on Nov 18, 2018

@author: liuyanan

https://www.youtube.com/watch?v=ZyTO4SwhSeE&list=PLQVvvaa0QuDfefDfXb9Yf0la1fPDKluPF&index=3
'''

import matplotlib.pyplot as plt
import csv
import numpy as np
import matplotlib.dates as mdatas


def bytespdata2num(fmt):
    strconverter = mdatas.strpdate2num(fmt)
    return strconverter


def basic_customization():

    # fig = plt.figure()
    ax1 = plt.subplot2grid((1, 1), (0, 0))

    date, close_price, high_price, low_price = np.loadtxt('601398.csv',
                                                          delimiter=',',
                                                          skiprows=1,
                                                          unpack=True,
                                                          encoding='gbk',
                                                          converters={0: bytespdata2num('%Y-%m-%d')},
                                                          usecols=(0, 3, 4, 5))

    ax1.plot_date(date, close_price, fmt='-', label='close_price')
#     ax1.plot_date(date, high_price, fmt='-', label='high_price')
#     ax1.plot_date(date, low_price, fmt='-', label='low_price')

    ax1.fill_between(date, close_price, close_price[0], where=(close_price > close_price[1]), facecolor='g', alpha=0.4)

    ax1.fill_between(date, close_price, close_price[0], where=(close_price <= close_price[1]), facecolor='r', alpha=0.4)

    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)
    ax1.grid(True)
    ax1.xaxis.label.set_color('c')
    ax1.yaxis.label.set_color('r')
    ax1.set_yticks([1, 2, 3, 4, 5, 6, 7])

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('ICBC')
    plt.legend()
    plt.subplots_adjust(left=0.09, bottom=0.16, right=0.94, top=0.90, wspace=0.9, hspace=0)
    plt.show()


def spines_and_horizontal_line():
        # fig = plt.figure()
    ax1 = plt.subplot2grid((1, 1), (0, 0))

    date, close_price, high_price, low_price = np.loadtxt('601398.csv',
                                                          delimiter=',',
                                                          skiprows=1,
                                                          unpack=True,
                                                          encoding='gbk',
                                                          converters={0: bytespdata2num('%Y-%m-%d')},
                                                          usecols=(0, 3, 4, 5))

    ax1.plot_date(date, close_price, fmt='-', label='close_price')
#     ax1.plot_date(date, high_price, fmt='-', label='high_price')
#     ax1.plot_date(date, low_price, fmt='-', label='low_price')

    ax1.axhline(close_price[0], color='k')
    ax1.axhline(close_price[10], color='k')
    ax1.axhline(close_price[20], color='k')

    ax1.fill_between(date, close_price, close_price[0], where=(close_price > close_price[1]), facecolor='g', alpha=0.4)
    ax1.fill_between(date, close_price, close_price[0], where=(close_price <= close_price[1]), facecolor='r', alpha=0.4)
    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)
    ax1.grid(True)
#     ax1.xaxis.label.set_color('c')
#     ax1.yaxis.label.set_color('r')
    ax1.set_yticks([1, 2, 3, 4, 5, 6, 7])

    ax1.spines['left'].set_color('c')
    ax1.spines['bottom'].set_color('r')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    ax1.spines['bottom'].set_linewidth(5)
    ax1.tick_params(axis='x', colors='y')
    ax1.tick_params(axis='y', colors='g')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('ICBC')
    plt.legend()
    plt.subplots_adjust(left=0.09, bottom=0.16, right=0.94, top=0.90, wspace=0.9, hspace=0)
    plt.show()


if __name__ == '__main__':
    # basic_customization()
    spines_and_horizontal_line()
