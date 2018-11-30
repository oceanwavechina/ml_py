# coding:utf-8
'''
Created on Nov 18, 2018

@author: liuyanan

https://www.youtube.com/watch?v=ZyTO4SwhSeE&list=PLQVvvaa0QuDfefDfXb9Yf0la1fPDKluPF&index=3
'''

import matplotlib.pyplot as plt
import csv
import numpy as np
import urllib
import matplotlib.dates as mdatas


def load_data_from_files():

    x = []
    y = []

    with open('example.txt', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            x.append(int(row[0]))
            y.append(int(row[1]))

    plt.plot(x, y, label='loaded from file')

    plt.xlabel('Plot Number')
    plt.ylabel('Important var')
    plt.title('Interesting Graph\nCheck it out')
    plt.legend()
    plt.show()


def load_data_from_files_with_numpy():

    x, y = np.loadtxt('example.txt', delimiter=',', unpack=True)

    plt.plot(x, y, label='loaded from file')

    plt.xlabel('Plot Number')
    plt.ylabel('Important var')
    plt.title('Interesting Graph\nCheck it out')
    plt.legend()
    plt.show()


def bytespdata2num(fmt):
    strconverter = mdatas.strpdate2num(fmt)
    return strconverter


def converting_data():

    date, close_price, high_price, low_price = np.loadtxt('601398.csv',
                                                          delimiter=',',
                                                          skiprows=1,
                                                          unpack=True,
                                                          encoding='gbk',
                                                          converters={0: bytespdata2num('%Y-%m-%d')},
                                                          usecols=(0, 3, 4, 5))

    plt.plot_date(date, close_price, fmt='-', label='close_price')
    plt.plot_date(date, high_price, fmt='-', label='high_price')
    plt.plot_date(date, low_price, fmt='-', label='low_price')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interesting Graph\nCheck it out')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # load_data_from_files()
    # load_data_from_files_with_numpy()
    converting_data()
